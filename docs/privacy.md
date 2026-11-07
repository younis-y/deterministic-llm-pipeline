# Privacy and storage

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
