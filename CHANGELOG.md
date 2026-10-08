# Changelog

All notable changes to rolescan. Newest first.

## 2.5.8

### Added
- `rolescan backup`: copies the store to `backups/` beside it with SQLite's backup API and checks the copy with `integrity_check`. `rolescan scan` takes the day's copy before it opens the store and keeps `output.backup_keep` of them (default 7). A copy that fails its check stops the scan with exit 1. `backup_keep: 0` turns off the automatic copy only: `rolescan backup` still copies on demand, and never deletes a copy.
- A run lock (`<db>.lock`): a second `scan` or `prune` on the same store stops at once, names the run that holds it, and exits 4.
- `output.retention_days` (`postings` 90, `deferred` 45, `verdicts` 180; 0 keeps a table whole). `rolescan prune` trims every cache, never `seen` or `applications`, and compacts the file when more than a fifth of it is free; a real `scan` prunes after it records. The verdicts age a scan or a bare `prune` uses is `max(retention_days.verdicts, llm.cache_days)`, so a verdict that is still a valid cache hit is never trimmed, and `llm.cache_days: 0` (a verdict never expires) means none is pruned. `rolescan prune --days N` is used as typed, whatever `cache_days` says.
- `llm.num_ctx` (default 12288), sent on every Ollama call; plugins build their options with `rolescan.scoring.judges.ollama_options`. An answer whose token counts show the prompt was cut, or the context window full, is an error, not facts. One retry after a timeout or a 5xx.
- `llm.check_truncation` (default true): the check above assumes `prompt_eval_count` counts the whole prompt even when Ollama reuses its cache, as the server this was built against does. Set it to false on a server that counts only the uncached part and every call would otherwise look cut; the counts are still recorded, never judged.
- The pre-scan check reads the model's own context window from Ollama's `/api/show` and reports an `llm.num_ctx` it cannot hold: the default of 12,288 refuses a model whose own window is 8k. No posting is sent to it, the run falls back to keyword scoring and `scan` exits 3, until `llm.num_ctx` is lowered to that window or a model with a longer one is chosen; `llm.check_truncation: false` does not lift this. The served model's digest is kept on `ScanResult.llm_model_digest` and printed in the digest's stats line. If the digest cannot be read (a failed probe, re-read once), the facts cache is skipped for that run, neither read nor written, and the digest says so.
- A circuit breaker: after five failed model calls in a row the scorer stops calling and defers the rest of the run ("after the LLM stopped answering"); postings with cached facts are still judged. A tripped breaker stops enrichment calls too, and enrichment failures do not count towards it.
- `rolescan scan` exits 3 when LLM scoring failed as a whole: the judge was configured and nothing was scored, the breaker tripped, or more than 20% of the postings that reached the model failed (a cache hit did not reach it). The digest is written, recorded and sent first. A failed email still exits 1, and that wins over 3.
- Sources say how much of their board they read: `Source.total` (the board's own count), `Source.truncated` (why the read stopped short) and `Source.note` (a line that is not a failure), copied onto `SourceReport`. The digest lists "Sources that shrank" (under 30% of the median of the last 14 days' non-zero runs, once there are three of them and the median is 10 or more), "Sources that were cut short" and "Notes", each even when it is the only thing to report. A board narrowed with Workday `applied_facets` or `search_text`, Adzuna `queries` or an `exclude_pattern` starts a fresh history, so narrowing it is not reported as a collapse. Entries that already use Adzuna `queries` or an `exclude_pattern` get a new history key at the 2.5.8 upgrade, so their quiet and shrink alarms start from a fresh baseline (three runs).
- Workday: `max_rows` (postings per pass, default 500, the old fixed cap), `applied_facets` (Workday's own filters) and `search_text` (one listing pass per text, each posting kept once). A board that holds more than was read is listed as cut short.
- `http.contact_url`: the User-Agent is `rolescan/<version> (+<contact_url>)`, the project's page by default; a blank one is a config error. `http.user_agent` still replaces it whole.
- Migration 8 adds `source_counts.total`, the board's own count beside what was read (0 when none was stated, or the board was empty; nothing reads it yet).

### Changed
- The facts cache holds the model's raw facts; verification and the resolvers run on every read, so a resolver change reaches cached postings with no model call. The key is `facts-v14` plus a fingerprint of the prompt, the candidate summary, the worked examples, the schema, the model digest, `num_ctx`, `description_chars` and `temperature`, so changing any of them asks the model again. Cached facts are re-extracted once.
- `Store.filter_new` looks postings up 500 at a time (SQLite before 3.32 refused more than 999 in one query) and, on a real scan, refreshes `seen.last_seen` for postings still listed.
- Migration 7 indexes `seen.url` (used by `unsee` and `mark`) and drops a duplicate `source_counts` index.
- Resolver: a range corrects a years value only when it is a range of years ("a team of 2-5; 5 years" stays 5), or follows a years label ("Experience required: 2-5", "Years of Experience Required: 1-3", "YOE: 3-5" are still read at their low end); "Our people bring N years" is not a requirement; "Security clearance: never required" is not a bar; soft, capped and excluded years ("would be an asset", "Even better if you have", "at most", "or less", "are not eligible", "During N years as ...") are not requirements; "Ensure that contractors hold security clearance" is not a bar.
- The resolvers' guards are pinned by a suite of probe sentences (`tests/test_resolver_guards.py`): disabling any whole guard fails a test (48 of 48), including `_BAR_CLAUSE_START`, which stays because it stops a soft word across a comma ("Saudi nationals only, Python is a plus.") from waiving a bar. The line-level mutation score of the two guarded resolvers rose from 22.6% to 51.6%.
- Keyword matching checks for a term as a substring before the whole-word pattern: about fifteen times faster, identical results.
- The digest stats line prints disjoint figures, "N scored, M from cache": a posting is in one or the other.
- `rolescan prune --days` takes 1 or more (`--days 0` used to drop every cached verdict; `retention_days.verdicts: 0` now means keep them all).
- The default User-Agent names the running version and the project page, not `rolescan/2.2 (personal job search tool)`.

### Fixed
- A store file that is not a database no longer leaves the process hanging after its traceback.
- Leaving `Store` through an exception rolls back instead of committing, so an interrupted `record_all` marks nothing seen.
- A store written by a newer rolescan is refused (`StoreTooNewError`) instead of opened. `scan`, `prune` and `backup` read its version first and exit 1 with the message, before the day's copy is taken, so the copy from before the upgrade is never rotated out.
- The digest stats line no longer prints "()" for a deferral reason it has no label for.
- One Workday posting that fails validation (a posting with no title) no longer fails the whole board: it is skipped and counted under "Notes".
- A Workday source read twice no longer reports the first read's note on the second.

### For library callers
- New: `ScanResult.llm_model`, `llm_model_digest`, `llm_scored`, `llm_breaker`, `llm_window_stop`, `facts_cache_skipped` and the `llm_failure` property; `FitScorer(..., model_digest=, facts_cache=, context_window=)`; `Store.filter_new(..., touch=)`; `Store.prune_all`, `PruneReport`, `StoreTooNewError`, `rolescan.store.refuse_a_newer_store`; `rolescan.storefile.run_lock`, `backup`, `RunLockedError`, `BackupError`; `rolescan.scoring.judges.backend_status`, `BackendStatus` (with `context_window`), `Judge.context_window`, `served_model_digest`, `PromptTruncatedError`, `ollama_options`; `rolescan.scoring.llm.finish_facts`, `prompt_fingerprint`, `FACTS_KEY_VERSION`; `Source.total`, `truncated`, `note`; `SourceReport.total`, `truncated`, `note`; `ScanResult.shrunk_sources`, `truncated_sources` and the `notes` property; `Store.source_counts_recent`; `rolescan.config.default_user_agent`, `DEFAULT_CONTACT_URL`, `HTTPConfig.contact_url`.
- `Store.record_source_counts` also takes `(count, total)` values; a plain count still works. `HTTPConfig.user_agent` defaults to "" and is filled from `contact_url` when the config is built.
- `cache_key` gains `fingerprint=`; `examples_digest=` is kept for compatibility and ignored when a fingerprint is given. Nothing was renamed or removed.

## 2.5.7

### Added
- A warning at load for a `title_only_blockers` entry with no `blockers` weight, and for a `hard_blockers` term that names a nationality listed in `profile.nationalities`.
- Digest stats line reports "N deferred to the next run" with the reason (over the LLM budget, over the digest cap, no description text yet), and "K listed as unread"; a deferred posting is not recorded as seen.
- "Hidden by your rules" gains "Your weighted terms (blockers)" and "Just under your keyword gate" groups.
- `profile.title_only_blockers`: `blockers` terms whose weight counts only in the job title.
- `profile.hidden_gate_margin`: how far under `min_keyword_score` a reject may fall and still be listed (default 10).
- `profile.nationalities`: nationalities you hold, so a matching nationality bar no longer blocks.
- `rules.field_exempt_companies`: employers whose postings pass the field rule.
- `rolescan unsee URL`: forget a posting so the next scan can report it again.
- `seen.reason` column (migration 5) records why a posting was hidden.
- `output.thin_unread_after` (default 3): a posting with no description text is deferred, then after that many runs listed once under "Unread (no text after N runs)" and recorded as seen (reason `thin_unread`), so a source that never sends text cannot defer it for ever. Counted in the new `deferred` table (migration 6); `--dry` never counts.
- `record_scan(cfg, result)`: the public half of a scan that writes `seen` once the digest exists.

### Changed
- Each scan writes its own digest file (`YYYY-MM-DDTHHMM.md`); `--dry` writes `digest-dry.md`.
- Postings are marked seen after the digest is written, not before. `run_scan` no longer writes `seen`; it returns `ScanResult.to_record` and library callers must call `record_scan(cfg, result)` after writing the digest (the CLI does). `--dry` writes `digest-dry.md` and leaves `latest.md` untouched.
- Resolver: students-only adverts with an alternative route are not skipped, a years range is read at its low end, a graduation year needs a graduation word, and promotion clauses are not read as requirements.
- Field rows: "transformation" reads as consulting, "Finance Project Analyst" as finance, and `other` never hides a graduate-entry role.
- Field rows widened (business/digital transformation = consulting; `<finance word> <word> analyst` = finance): for a profile whose `allowed_fields` excludes those, such titles are now hidden under Field (listed in Hidden by your rules).
- The quiet-source alarm looks back 14 days and always reaches the Markdown digest.
- Facts cache key bumped to `facts-v13`: cached facts are re-extracted once.

### Fixed
- A reject pushed under the keyword gate by weighted penalties while an unweighted `hard_blockers` term also matched was listed nowhere; it is now listed under "Your blocking terms".
- Migrations run inside one transaction with their `user_version` bump, and migration 5 skips a column that already exists, so two processes opening an old store at once, or a crash mid-upgrade, no longer leave it unopenable.
- Company matching for `rules.field_exempt_companies` ignores accents ("Societe Generale" matches "Société Générale").
- Adzuna now raises when every query fails, instead of reporting ok with nothing.
- `config.example.yaml` and `examples/energy-trading.yaml` no longer carry the removed `cv_dir` key, and a test now loads every shipped YAML.
