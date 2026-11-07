# Providers

A backend sends one item to a model and returns the model's raw answer.
Checking quotes, running resolvers and deciding happen afterwards in the
pipeline, the same for every backend, so changing provider never changes how
an answer is judged.

All backends read the `model:` block of the config:

| Key | Default | Meaning |
|---|---|---|
| `backend` | `anthropic` | `anthropic`, `openai-compatible`, or a plugin's name |
| `name` | required | the model, as the provider names it |
| `base_url` | none | the server's address (`openai-compatible`) |
| `api_key_env` | the backend's own | the environment variable that holds the key |
| `max_calls` | 100 | model calls per run; the rest are deferred |
| `max_minutes` | 30 | wall-clock minutes per run; the rest are deferred |
| `max_concurrent` | 4 | calls in flight at once |
| `max_tokens` | 2048 | the longest answer asked for |
| `temperature` | not sent | sent only when set |

## anthropic

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
```

```yaml
model: {backend: anthropic, name: claude-haiku-5-5}
```

The facts schema goes to the API as structured output, so an answer arrives
already in the schema's shape. The system prompt is sent as a cached block;
a prompt shorter than the model's minimum cacheable length is simply not
cached. Leave `temperature` unset for Claude: current models refuse it. If it
is set and the API refuses it, factgate logs that once and stops sending it.

The live results in the README were recorded with `claude-haiku-5-5`. The
three examples' 174 calls cost at most USD 0.05. Input token counts include
the prompt read back from the cache, which bills for less.

### Batch calls

```yaml
model: {backend: anthropic, name: claude-haiku-5-5, batch: true, max_minutes: 60}
```

With `batch: true` a run sends its items, and a live eval its attempts, as one
Message Batch, which bills every token at half the standard price. The first
prompt goes as an ordinary call that writes the shared prompt to a one-hour
cache entry, and the batch's requests read it. Requests in a batch run side by
side, so without that call each would write the prompt itself, and a write
costs more than the batch saves.

The run waits for the batch, checking every 10 seconds, until `max_minutes`.
Then it cancels it: what the batch answered is kept, and the rest is deferred
to the next run (in a live eval, it is inconclusive). A failed check counts as
"not yet", and a run stopped while it waits cancels its batch. `max_calls`
still caps how many items are sent.

Checked on 7 November 2026 with `claude-haiku-5-5` on the README's 58 cases,
3 attempts each. No case moved between pass and fail, and the one case that
was flaky with direct calls passed every attempt; the comparisons are in each
example's `results/batch/`. The support and tender batches (42 and 36
requests) took 6 and 5 minutes, and the invoice batch (96 requests) took 45.
So set `max_minutes` to as long as you can wait, and batch the runs nobody is
watching.

## openai-compatible

One backend for any server that speaks the OpenAI chat-completions API with
JSON-schema output. It posts to `{base_url}/chat/completions` with the facts
schema as `response_format`, sends `Authorization: Bearer <key>` when
`api_key_env` names a variable that is set, and retries once, after 2 seconds,
on a timeout or a 5xx.

**Ollama**, local, no key:

```yaml
model: {backend: openai-compatible, name: "qwen2.5:14b", base_url: "http://localhost:11434/v1"}
```

Checked on 7 November 2026 against Ollama 0.34.3 with `llama3.2:3b`: the
support-tickets example's 8 tickets were all decided, with no errors, in 14
seconds. That run checks the connection, not quality: the 3B model sent 3 of
the 8 tickets to the default outcome where Claude Haiku sent 1. Use a larger
model for real work, and measure it with a live eval before you trust it.

**vLLM, LM Studio, OpenAI and others**, by the same pattern (not tested here):

```yaml
model: {backend: openai-compatible, name: my-model, base_url: "http://localhost:8000/v1"}
model: {backend: openai-compatible, name: your-model, base_url: "https://api.openai.com/v1", api_key_env: OPENAI_API_KEY}
```

factgate sends `max_tokens`. OpenAI's reasoning models expect
`max_completion_tokens` instead, so they may refuse the request; check one
call before you rely on them.

## replay

Answers recorded earlier, keyed by item id. Record them with any backend:

```bash
factgate run config.yaml --record answers.jsonl
factgate run config.yaml --replay answers.jsonl
```

Each line of the file is `{"id": ..., "raw": <the model's answer>}`, written as
each answer arrives, answers from the cache and answers that do not fit the
schema included. An item with no recorded answer ends as `error`; it is not a
call, and it does not count towards the three errors in a row that stop a run. Replay is how the examples run with no
key, and how you reproduce a run exactly, since the cache is not involved. A
live eval keeps every answer in its own result file instead, and
`factgate eval regrade` re-scores those.

## The cache

Answers are kept in `.factgate/cache.sqlite` (the `cache:` key, relative to
the config; `cache: null` turns it off, `--no-cache` skips it for one run). A
row's key names the item's text in field order, the backend, the model, and a
fingerprint of the prompt, the schema, `max_tokens`, `temperature` and the
server's `base_url`, so changing any of them asks the model again rather than
serving an old answer. The quote check,
resolvers and rules run on every read, so a rule change reaches cached items
with no calls.

## Writing a backend

Subclass `Backend` and return an `Answer`:

```python
from pydantic import BaseModel

from factgate.backends import Answer, Backend


class MyBackend(Backend):
    name = "mine"

    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        raw = await call_my_model(system, user, schema=model.model_json_schema())
        return Answer(raw=raw, input_tokens=0, output_tokens=0)
```

Register it in your package's `pyproject.toml`, then name it in a config with
`backend: mine`:

```toml
[project.entry-points."factgate.backends"]
mine = "mypkg.backend:MyBackend"
```

`extract` raises on failure: the run counts it as an error for that item, and
three in a row stop the run and defer the rest. It must not check quotes or
decide anything; the pipeline does both.

A backend that can send many prompts at a lower price sets `batches = True`
and implements `extract_many(system, users, model, deadline)`. It returns one
entry per prompt, in order: an `Answer`, the exception that request failed
with, or `None` when the request was not answered by `deadline` (a
`time.monotonic()` value). The run defers a `None` and records an exception as
an error.

## Writing a reader

Items can come from your own reader too: a function that takes the `items:`
block and the config's folder and returns a list of `Item`, registered under
`factgate.sources`:

```toml
[project.entry-points."factgate.sources"]
tickets = "mypkg.readers:read_tickets"
```

```yaml
items: {plugin: tickets, options: {queue: billing}}
```
