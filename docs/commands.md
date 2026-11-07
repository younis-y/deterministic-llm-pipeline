# Commands

The list is in the [README](../README.md#commands). This page covers what
the less obvious ones do.

`mark` takes one of three states. `shortlist` keeps a posting in the digest's
**Shortlist** section, which is repeated under the new roles on every run so an
open application stays in front of you rather than scrolling away with
yesterday's email. `applied` clears it from that section. `dismissed` clears it
too, and additionally stops the posting ever being reported again, however many
boards go on listing it. Every shortlist row in the digest carries the exact
`mark` command to clear it, with the config path already filled in.

`scan --dry` marks nothing seen and writes `digest-dry.md`, leaving `latest.md`
as the last real run's digest. `unsee` takes a posting's URL (as the digest
prints it) or its uid and removes it from the seen list, so the next scan can
report it again.

`rolescan scan` tells a scheduler how the run went. Exit status: 0 ok; 1
email failed, the day's backup failed its check, or the store was written by
a newer rolescan (takes precedence over 3); 2 config error, or
http.bind_interface names an interface with no IPv4 address; 3 LLM scoring
failed as a whole (after the digest was written and sent); 4 another run
holds the store's lock. The digest says why. The store gets a checked copy in
`backups/` beside it once a day (`output.backup_keep`, default 7), and its
caches are trimmed after each real scan (`output.retention_days`).

If the digest is written but the email times out, a VPN may be blocking the mail
ports: set `output.email.bind_interface` to the interface that reaches the
internet directly (for example `en0`) and the mail socket binds to that
interface's address, looked up at every send (port 465 uses implicit TLS, any
other port STARTTLS).

The same VPN's exit address can be the thing a job site refuses, while the site
answers the same request from your own network. Set `http.bind_interface` to the
interface that reaches the internet directly (for example `en0`) and the
job-fetching connections leave from that interface's IPv4 address, read once
when a scan or `discover` starts. An interface with no IPv4 address stops the
run before any request, with exit status 2. The model backends are never bound,
so a model on localhost still answers. A site reachable over IPv6 only cannot be
reached from an IPv4 address, and `HTTP_PROXY` and `HTTPS_PROXY` are not used
while the setting is on.

`discover` distinguishes five outcomes on purpose. `EMPTY` means a real board
with no openings; `UNKNOWN` means an API that cannot tell an empty board from a
wrong slug. Collapsing those into a single "OK" hides which of them actually returned postings.

## Finding slugs

`rolescan slugs` searches a harvested ATS directory you download yourself. It
matches on `difflib.SequenceMatcher` with a length-aware cap and a directional
containment bonus, because absolute name length rather than length ratio is
what separates `octoenergy` from `octopusenergy` without also collapsing `vitl`
into `vitol`.

The dataset has no SmartRecruiters file and covers no bespoke platforms, so
expect to resolve some employers by hand: open the careers page, follow where it
redirects, and read the ATS and slug out of the final URL. `discover` then
confirms or refutes it.
