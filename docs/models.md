# Model cost and health

The three scoring levels, and the one measured cost, are in the README's
[Models, keys and cost](../README.md#models-keys-and-cost).

What keeps a model's cost down:

- Only postings that clear the keyword prefilter reach a model.
- A model's answer is cached for `llm.cache_days` (30), so an unchanged
  posting is not paid for twice.
- `llm.max_calls_per_run` (60) is a hard ceiling per scan. Postings past it are
  deferred to the next run, not dropped.
- `llm.max_minutes` (0, no limit) is a wall-clock budget for the model, counted
  from the scan's first model call. On a local model one call takes 10 to 40
  seconds, so a call count says little about how long a scan runs. When the
  budget is spent no new call starts, calls already running finish, and the
  postings not yet scored are deferred to the next run exactly as the call
  ceiling defers them: kept out of `seen`, so they come round again. The stats
  line says which limit stopped the run: "model time budget of 10 minutes
  spent; 12 postings deferred to the next run", or "(N over the LLM budget)"
  for the call count. Whichever was reached first is the one named.
- `llm.batch` (off) sends facts-mode calls on the `anthropic` backend through
  the Message Batches API, which bills every token at half the standard price.
  The scan waits for the batch to end, usually within the hour, so it suits a
  scan nobody is watching. One direct call goes first and writes the shared
  prompt to a one-hour cache entry, so the batch reads it instead of writing
  it. If the batch has not ended after `llm.batch_max_minutes` (120) it is
  cancelled: what it answered is kept, and the rest are scored on the next
  run.
- A call sends the fixed instructions, your `profile.summary`, any worked
  examples, and one posting's title, company, location, date and the first
  `llm.description_chars` characters of its description (6,000 by default): a
  few thousand tokens in, a few hundred out.

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
