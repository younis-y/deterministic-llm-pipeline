# Configuration reference

Generated from the config models by `scripts/gen_config_doc.py`. Do not edit it
by hand: a test fails when it differs from what the script prints. Copy
`config.example.yaml` to `config.yaml` to start, or `examples/quickstart.yaml`
for the smallest working file.

Unknown keys are an error (the config says which), except the per-source
options under `sources`. Relative paths resolve against the config file, not
the working directory. Anything left blank that names a secret is read from the
environment.

## How terms are matched

Three matchers are in play, and they differ on purpose.

- `profile.keywords` are plain, case-insensitive **substrings**. `ai` matches
  `maintain`, `email` and `said`. Write the whole phrase.
- `profile.blockers`, `profile.hard_blockers`, `profile.title_only_blockers`,
  `profile.excluded_locations`, `profile.agencies` and
  `profile.rules.field_exempt_companies` match **whole words**, ignoring case:
  `director` does not match `directorate`. A blocker can delete a role, so it
  has to mean the word.
- `profile.locations` is a case-insensitive substring of the posting's location
  field.

## profile

Who you are and what counts as a good role.

| Key | Type | Default | What it does |
|---|---|---|---|
| `profile.name` | text | empty | A label for yourself. Nothing reads it. |
| `profile.summary` | text | empty | Your background in your own words. Sent as it is to a model backend as the candidate; keyword-only scoring never reads it. Numbers beat adjectives. |
| `profile.locations` | list of text | empty | Places you will work. A posting whose location contains none of these (a case-insensitive substring) is charged `location_penalty`. A posting with no stated location is not penalised. Empty turns the check off. |
| `profile.allow_remote` | true or false | `true` | A remote posting is never charged `location_penalty`. |
| `profile.keywords` | map of text to whole number | empty | Weighted terms. Each is matched as a plain, case-insensitive SUBSTRING of the posting text, not a whole word, so a short term matches inside longer ones: `ai` hits `maintain` and `email`. Write `machine learning` or `ai engineer`, not `ai`. A term found in the title counts three times its weight; a term counts once however often it occurs. |
| `profile.blockers` | map of text to whole number | empty | Terms that cost points. Each is a case-insensitive WHOLE-WORD match (`director` does not hit `directorate`) and subtracts its weight from the keyword score. A posting pushed under `min_keyword_score` is dropped before scoring and recorded as seen. A preference, not a bar: see `hard_blockers`. |
| `profile.hard_blockers` | list of text | empty | Structural bars: things no application can get past, such as a clearance or a nationality gate. A whole-word, case-insensitive match marks the posting `blocked` whatever a model says. Keep it short and specific: with `output.show_blocked: false` a blocked posting is removed and never shown again. An empty or one-character entry is a config error. |
| `profile.title_only_blockers` | list of text | empty | `blockers` terms whose weight counts only when the term is in the job title, for words such as `head of` that are boilerplate in a body (`head office`). A term here needs a weight in `blockers`. One also in `hard_blockers` still bars on the whole text. |
| `profile.hidden_gate_margin` | whole number, 0 or more | `10` | How far under `min_keyword_score` a posting may fall and still be listed under "Hidden by your rules" in the digest, so a weight that rejects good roles shows up. `0` lists none. |
| `profile.location_penalty` | whole number | `25` | Points subtracted when the location check fails. |
| `profile.min_keyword_score` | whole number, 0 or more | `18` | The prefilter gate. A posting scoring below this never reaches a model, which makes it the biggest lever on cost. |
| `profile.excluded_locations` | list of text | empty | Places you cannot or will not work. A whole-word, case-insensitive match against the posting's LOCATION only, never its description. A match blocks the posting like a `hard_blockers` term. |
| `profile.nationalities` | list of text | empty | Nationalities you hold, in the words an advert uses: the demonym and the country. A nationality bar whose quote names one is not a bar for you. Acts in facts mode only. A matching term in `hard_blockers` still blocks, so remove it from there. |
| `profile.agencies` | list of text | empty | Company names that post roles they are not hiring for: recruiters, staffing firms. A whole-word match against the COMPANY name only. Does nothing unless `agency_penalty` is above 0. |
| `profile.agency_penalty` | whole number, 0 or more | `0` | Points an agency listing loses. `0` turns it off. A penalty and not a bar, so an agency-only role still appears while the employer's own posting ranks above it. |
| `profile.max_age_days` | whole number, 0 or more | `90` | Drop postings older than this many days, from sources whose dates mean freshness (aggregators). Employer boards are exempt, and so is a posting with no date. `0` turns it off. |
| `profile.min_report_score` | whole number, 0 to 100 | `55` | The final gate: a posting scoring below this is not in the digest. Model scores and keyword-only scores are on different scales, and a keyword-only run wants a lower number (about 20 to 30). |
| `profile.rules` | section | see below | Eligibility rules, applied by code in facts mode to the facts a model quotes from each posting. Omit them and no rule fires. See `profile.rules` below. |

### profile.rules

Eligibility rules for facts mode. Each fires only on a fact the advert states, and every skip names the rule and quotes the advert.

| Key | Type | Default | What it does |
|---|---|---|---|
| `profile.rules.max_years_required` | whole number, 0 or more | unset | Skip a posting that states a minimum above this many years. A posting that states no years is never skipped by it. |
| `profile.rules.allowed_levels` | list of choices | `graduate_entry`, `junior`, `mid`, `not_stated` | Levels that pass the level rule. Keep `not_stated`, or every advert that states no level is skipped. Values: `graduate_entry`, `junior`, `mid`, `senior`, `lead_principal`, `not_stated`. |
| `profile.rules.student_only` | `"skip"` or `"allow"` | `allow` | `skip` drops roles open only to current students; `allow` keeps them. |
| `profile.rules.allowed_fields` | list of choices | unset | Fields that pass the field rule; unset means any field. The field is read from the title first. Values: `data_engineering`, `ai_llm`, `data_science`, `analytics_bi`, `quant`, `product`, `software`, `consulting`, `finance`, `other`. |
| `profile.rules.max_graduation_year` | whole number | unset | Skip a stated graduation year above this, the earliest year the advert accepts. Unset is no limit. While set, `student_only` applies only to adverts that state no year. |
| `profile.rules.level_from_title_only` | true or false | `false` | `true`: only a level word in the job title can fire the level rule, and a level the model states is ignored. |
| `profile.rules.field_exempt_companies` | list of text | empty | Employers whose postings pass the field rule whatever their field: a company name, matched as whole words, ignoring case and accents. |

## llm

How postings are scored, and by what.

| Key | Type | Default | What it does |
|---|---|---|---|
| `llm.enabled` | true or false | `true` | `false` scores on keywords alone: no key, no network, no cost. Also switched off, with a note in the digest, when a hosted backend has no key. |
| `llm.backend` | text | `anthropic` | Which plugin scores postings: `anthropic` (needs `ANTHROPIC_API_KEY`) or `ollama` (a local model, no key). `rolescan backends` lists what is registered. |
| `llm.base_url` | text | `http://localhost:11434` | Where a local backend listens. Ollama's default listener. |
| `llm.timeout` | number | `120.0` | Seconds per posting for a local model, which can be slow. |
| `llm.model` | text | `claude-sonnet-5` | The model the backend is asked for. For `ollama`, one you have pulled. |
| `llm.api_key` | text | from the environment | The key for a hosted backend. Leave it blank to read the backend's environment variable (`ANTHROPIC_API_KEY`). If you write a key here, keep this file out of version control. |
| `llm.max_tokens` | whole number | `1500` | The longest answer asked of the model, in tokens. |
| `llm.max_concurrent` | whole number, 1 to 32 | `5` | Model calls in flight at once. |
| `llm.max_calls_per_run` | whole number, 0 or more | `60` | Hard ceiling on model calls in one scan. Postings past it are deferred to the next run, not dropped. `0` allows no calls. |
| `llm.cascade` | true or false | `true` | Score in two passes: a short call that settles the score, then the full verdict only for postings that clear `min_report_score`. Used only by a backend whose short call is cheaper. |
| `llm.extra_prompt` | text | empty | Text appended to the judge-mode prompt. Not sent in facts mode, where only an enricher may use it. |
| `llm.temperature` | number, 0.0 to 2.0 | `0.0` | Sampling temperature. `0` suits classification against a fixed rubric. A model that refuses the setting is called without it from then on, and the log says so once. |
| `llm.description_chars` | whole number, 500 or more | `6000` | How many characters of a posting's description are sent to a model. |
| `llm.num_ctx` | whole number, 2048 or more | `12288` | Context window, in tokens, asked of a local (Ollama) model on every call. rolescan checks before the scan that the model can hold it, and refuses a model whose own window is smaller. |
| `llm.check_truncation` | true or false | `true` | Treat an answer whose token counts show a cut prompt as an error. Set `false` only for an Ollama that counts just the uncached part of a prompt. |
| `llm.cache_days` | whole number, 0 or more | `30` | Days a cached model answer stays valid. `0` never expires. |
| `llm.mode` | `"facts"` or `"judge"` | `facts` | `facts` (default): the model extracts quoted facts and code applies `profile.rules`. `judge`: one call per posting, the model gives the score, verdict and blockers itself, `profile.rules` do not apply and `llm.extra_prompt` is used. A supported mode: see the README, "Judge mode". |
| `llm.enricher` | text | empty | Name of a registered enricher (entry-point group `rolescan.enrichers`) that does extra work only for postings clearing `min_report_score`. Empty means none. A failing enricher keeps the plain verdict. |
| `llm.facts_examples_file` | path | unset | A YAML list of worked examples (title, company, description, facts) added to the facts-mode prompt. Relative to this file, and validated when the config loads. Unset means no examples. |

## http

How requests are made.

| Key | Type | Default | What it does |
|---|---|---|---|
| `http.timeout` | number | `20.0` | Seconds allowed per request. |
| `http.max_concurrent` | whole number, 1 to 64 | `8` | Requests in flight at once. |
| `http.max_retries` | whole number, 0 to 10 | `3` | Retries per failed request. |
| `http.contact_url` | text | the project's page | Shown in the User-Agent, `rolescan/<version> (+<contact_url>)`, so a site owner can see what is calling. Point it at your fork or a page of your own. A blank value is a config error. |
| `http.user_agent` | text | `rolescan/<version> (+<contact_url>)` | Replaces the whole User-Agent header when set. |

## output

Where results go, and how long caches live.

| Key | Type | Default | What it does |
|---|---|---|---|
| `output.dir` | path | `digests` | Where digests are written, relative to this file. |
| `output.db_path` | path | `seen.db` | The store (SQLite), relative to this file. |
| `output.max_roles` | whole number, 1 or more | `15` | Most roles in one digest. The rest are deferred to the next run, not dropped. |
| `output.show_blocked` | true or false | `true` | List blocked roles at the bottom of the digest. `false` removes them, and they are recorded as seen. |
| `output.hidden_max` | whole number, 0 or more | `60` | Most lines in the digest's "Hidden by your rules" section, counted across every rule, highest scores first; each rule's count is shown either way. The rest are written to a `-hidden.md` file beside the digest, which names it. In the email the roles get the room first and this section gets what they leave (a line costs about 0.3 KB, and Gmail clips a message past about 102 KB), so a high value never pushes a role out; the email lists as many as fit, and the Markdown digest lists up to this many. `0` lists none. |
| `output.thin_unread_after` | whole number, 1 or more | `3` | Runs a posting may arrive without text before it is listed once under "Unread" and recorded. |
| `output.retention_days` | section | see below | How many days each cache keeps a row. See `output.retention_days` below. |
| `output.backup_keep` | whole number, 0 or more | `7` | Daily copies of the store kept in `backups/` beside it. `0` turns the automatic copy off; `rolescan backup` still copies on demand. |
| `output.email` | section | see below | Sending the digest by email. See `output.email` below. |

### output.retention_days

Days a cache keeps a row. Never touches what has been seen, or what you marked.

| Key | Type | Default | What it does |
|---|---|---|---|
| `output.retention_days.postings` | whole number, 0 or more | `90` | Cached posting pages. A trimmed page costs one re-fetch. `0` keeps them all. |
| `output.retention_days.deferred` | whole number, 0 or more | `45` | Deferral counts of postings not sighted for this long. `0` keeps them all. |
| `output.retention_days.verdicts` | whole number, 0 or more | `180` | Cached model answers. Never less than `llm.cache_days`. `0` keeps them all. |

### output.email

Emailing the digest.

| Key | Type | Default | What it does |
|---|---|---|---|
| `output.email.enabled` | true or false | `false` | Send the digest by email. Switched off, with a note, when `smtp_host` or the password is missing. |
| `output.email.smtp_host` | text | empty | The mail server. |
| `output.email.smtp_port` | whole number | `587` | `587` connects in plain text and upgrades with STARTTLS; `465` uses implicit TLS, encrypted from the first byte. Either way the server's certificate must verify against your system's trusted authorities; a relay with a self-signed certificate is refused. |
| `output.email.username` | text | empty | The login name on the mail server. |
| `output.email.password` | text | from `ROLESCAN_SMTP_PASS` | The mail password. Leave it blank and `ROLESCAN_SMTP_PASS` is read. |
| `output.email.to` | text | empty | Where the digest is sent. |
| `output.email.bind_interface` | text | empty | The network interface whose IPv4 address the mail socket binds to (`en0`, `eth0`), for a machine where a VPN blocks the mail ports while HTTPS passes. Looked up at every send. Empty leaves it to the default route. |

## sources

A list. Each entry names one board. Beyond the keys below, an entry carries the options of its kind; any other key is kept and passed to the source, so a misspelt option is silently ignored: check it with `rolescan discover`.

| Key | Type | Default | What it does |
|---|---|---|---|
| `sources[].kind` | text | required | A registered source kind. `rolescan sources` lists them with their slug formats. |
| `sources[].slug` | text | required | The employer's board identifier; its format depends on the kind. |
| `sources[].label` | text | empty | The name shown in the digest. Defaults to the slug. |
| `sources[].enabled` | true or false | `true` | `false` keeps the entry in the file but skips it; `discover` marks it disabled. |
| `sources[].verified` | true or false | `false` | A note to yourself that `discover` confirmed the entry. Nothing reads it. |

### Options by kind

| Kind | Option | Default | What it does |
|---|---|---|---|
| `workday` | `site` | the slug | The site name in the careers URL. |
| `workday` | `host` | `wd3` | The Workday host number in the careers URL. |
| `workday` | `details` | `true` | Fetch each posting's description (one extra request each). `false` lists titles only. |
| `workday` | `max_rows` | `500` | Postings read per listing pass. A board holding more is reported as cut short. |
| `workday` | `applied_facets` | none | Workday's own filters: facet parameter to a list of value ids, taken from the board's listing answer. |
| `workday` | `search_text` | none | One listing pass per text (a string or a list); each posting is kept once. |
| `structured` | `sitemap` | none | The sitemap URL. Required. |
| `structured` | `url_pattern` | `/job` | A regex; only sitemap URLs it matches are fetched. |
| `structured` | `exclude_pattern` | none | A regex; matching URLs are dropped before any fetch, for a board that mixes regions. |
| `structured` | `delay` | `0.5` | Seconds between detail-page requests. A `Crawl-delay` in the host's robots.txt raises it to that (never past 30 seconds). |
| `structured` | `max_pages` | `500` | Most detail pages read per run. A sitemap with more in scope is reported as cut short. |
| `structured` | `max_age_days` | unset | Skip a URL whose sitemap `lastmod` is more than this many days old, before any fetch. A URL with no readable `lastmod` is kept. |
| `structured` | `incremental` | `false` | `true` reads only the URLs whose `lastmod` is newer than the last scan that read its whole window, newest first, up to `max_pages`; a scan with no mark reads everything in scope up to the same cap. The mark moves only after a real scan that was recorded with nothing deferred, no model failure, no read cut short at `max_pages` and no page that failed for now. Otherwise the next scan opens the same window again, so a cut read repeats until `max_pages` or `max_age_days` covers the window. It compares the sitemap's `lastmod` and nothing else, so a URL first listed with a `lastmod` older than the mark (a posting the board published late, or under an old date) is not read; set it `false` for one scan to sweep the whole window. Its counts vary by design, so the quiet and shrink alarms skip it. A `--dry` run keeps no mark. To read everything once more, set it `false` for a scan. |
| `structured` | `max_sitemap_urls` | `50000` | Refuse a sitemap that lists more URLs than this, with an error that says how to narrow the read. |
| `adzuna` | `app_id` | from `ADZUNA_APP_ID` | Your Adzuna application id. Leave it blank to read the environment. |
| `adzuna` | `app_key` | from `ADZUNA_APP_KEY` | Your Adzuna application key. Leave it blank to read the environment. |
| `adzuna` | `queries` | none | A list of search texts, one search each. Required. |
| `adzuna` | `max_days_old` | `7` | Only postings no older than this. |
| `adzuna` | `where` | none | A place to search near. |
| `adzuna` | `max_pages` | `5` | Pages read per query. A query with more results than `max_pages` x `results_per_page` is reported as cut short (with the API's own count when the entry has one query). |
| `adzuna` | `results_per_page` | `50` | Rows a page asks for, 1 to 50 (Adzuna's own limit). A page with fewer rows is the last one. |
| `reed` | `api_key` | from `REED_API_KEY` | Your Reed API key. Leave it blank to read the environment. With neither, the source is skipped, and the digest says so. |
| `reed` | `queries` | none | A list of search texts, one search each. Required. |
| `reed` | `where` | none | A place to search near (Reed's `locationName`). |
| `reed` | `distance` | unset | Miles around `where`. Reed's own default applies. |
| `reed` | `graduate` | `false` | `true` asks for graduate roles only. |
| `reed` | `direct_employer_only` | `false` | `true` leaves out roles posted by a recruitment agency. |
| `reed` | `max_pages` | `5` | Pages read per query. A query with more results than `max_pages` x `results_per_page` is reported as cut short (with Reed's own count when the entry has one query). |
| `reed` | `results_per_page` | `100` | Rows a page asks for, 1 to 100 (Reed's own limit). A page with fewer rows is the last one. |
| `jooble` | `api_key` | from `JOOBLE_API_KEY` | Your Jooble API key. Leave it blank to read the environment. With neither, the source is skipped, and the digest says so. |
| `jooble` | `queries` | none | A list of search texts, one search each. Required. |
| `jooble` | `where` | none | A place to search (Jooble's `location`). |
| `jooble` | `radius` | unset | Kilometres around `where`. |
| `jooble` | `salary` | unset | The least pay wanted, as Jooble reads it. |
| `jooble` | `max_pages` | `5` | Pages read per query. Jooble sets the page size, so a query is read until a page comes back empty, its own count is reached, or this many pages are read; a query cut at this is reported as cut short. |
| `workable_search` | `queries` | none | A list of search texts, one search each. Required. |
| `workable_search` | `where` | none | A place to search (Workable's `location`). |
| `workable_search` | `max_pages` | `5` | Pages read per query. Workable sets the page size, so a query is read until no further page is offered, its own count is reached, or this many pages are read; a query cut at this is reported as cut short. |
| `workable_search` | `delay` | `2` | Seconds between requests. Queries run one after another, and pages within a query too, a `delay` apart, never together. `0` turns the wait off. A read also stops, and says why in the digest, when a page repeats rows already read or Workable gives back a page token it was already sent, which means the paging parameter is not being honoured. |

`greenhouse`, `lever`, `ashby`, `workable` and `smartrecruiters` take only the keys above.

## Environment variables

| Variable | Read when |
|---|---|
| `ANTHROPIC_API_KEY` | `llm.api_key` is blank and the backend is `anthropic` |
| `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` | an `adzuna` entry has no `app_id` or `app_key` |
| `REED_API_KEY` | a `reed` entry has no `api_key` |
| `JOOBLE_API_KEY` | a `jooble` entry has no `api_key` |
| `ROLESCAN_SMTP_PASS` | `output.email.password` is blank |

rolescan does not read a `.env` file: export the variable in your shell or scheduler.
