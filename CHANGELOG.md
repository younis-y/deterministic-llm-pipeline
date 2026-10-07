# Changelog

All notable changes to rolescan. Newest first.

## 2.5.7

### Added
- Digest stats line reports "N deferred to the next run" with the reason (over the LLM budget, over the digest cap, no description text yet); a deferred posting is not recorded as seen.
- "Hidden by your rules" gains "Your weighted terms (blockers)" and "Just under your keyword gate" groups.
- `profile.title_only_blockers`: `blockers` terms whose weight counts only in the job title.
- `profile.hidden_gate_margin`: how far under `min_keyword_score` a reject may fall and still be listed (default 10).
- `profile.nationalities`: nationalities you hold, so a matching nationality bar no longer blocks.
- `rules.field_exempt_companies`: employers whose postings pass the field rule.
- `rolescan unsee URL`: forget a posting so the next scan can report it again.
- `seen.reason` column (migration 5) records why a posting was hidden.

### Changed
- Each scan writes its own digest file (`YYYY-MM-DDTHHMM.md`); `--dry` writes `digest-dry.md`.
- Postings are marked seen after the digest is written, not before.
- Resolver: students-only adverts with an alternative route are not skipped, a years range is read at its low end, a graduation year needs a graduation word, and promotion clauses are not read as requirements.
- Field rows: "transformation" reads as consulting, "Finance Project Analyst" as finance, and `other` never hides a graduate-entry role.
- The quiet-source alarm looks back 14 days and always reaches the Markdown digest.
- Facts cache key bumped to `facts-v13`: cached facts are re-extracted once.

### Fixed
- Adzuna now raises when every query fails, instead of reporting ok with nothing.
- `config.example.yaml` and `examples/energy-trading.yaml` no longer carry the removed `cv_dir` key, and a test now loads every shipped YAML.
