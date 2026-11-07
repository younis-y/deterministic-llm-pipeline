# What gets hidden, and where to look

rolescan leaves postings out of your digest, and it reports most of what it
leaves out. Where to look for each kind:

| Why a posting is not in your digest | Where to look |
|---|---|
| Its keyword score was under `min_keyword_score`, so no model saw it | The stats line counts it ("N filtered before scoring"). It is not listed. To see more, lower the number and re-assess (see below). |
| Its score was under `min_report_score` | The stats line counts it ("N below min_report_score"). It is not listed. Lower the number and re-assess (see below). |
| A `profile.rules` rule skipped it, or the advert states a nationality or clearance bar | **Hidden by your rules**, one line each: the rule, and the advert's own words. |
| A `hard_blockers` term or an `excluded_locations` entry matched | Shown at the bottom, marked BLOCKED, while `output.show_blocked` is `true` (the default). With `false` it is removed and listed under **Hidden by your rules**, and the stats line says "N blocked and hidden". |
| Weighted `blockers` pushed it under the keyword gate, or it fell within `profile.hidden_gate_margin` of the gate | **Hidden by your rules**, under "Your weighted terms" or "Just under your keyword gate". |
| It is waiting its turn (model budget, digest cap, no text yet) | "N deferred to the next run" in the stats line. A deferred posting is not recorded as seen, so it comes round again. |

The **Hidden by your rules** section appears only when something was hidden, and
the stats line then says "N hidden by your rules (listed below)". It is how you
catch a rule or a term that matches the wrong thing. Each group heading carries
its count. It looks like this:

```markdown
## Hidden by your rules

Skipped or blocked by one of your rules, so not listed above. One line each, so a wrong skip can be spotted.

**Years of experience** (1)

- **Example Co** · [Senior Data Engineer](https://example.test/jobs/12) · London: advert asks for "5+ years"

**Your blocking terms (hard_blockers, excluded_locations)** (1)

- **Example Plc** · [Data Analyst](https://example.test/jobs/31) · Leeds: blocked by "security clearance"
```

**At volume, the section is capped and the alarms come first.** The digest lists
at most `output.hidden_max` hidden postings (60 by default), the highest-scoring
first, since a hide with a high score is the likeliest wrong one. Every rule's
count is still shown. When the cap cuts the list, the whole of it is written to
`<digest name>-hidden.md` beside the digest, the digest names that file, and the
stats line reads "N hidden by your rules (M listed below, all N in the file)".
Gmail clips an email past about 102 KB, so the digest opens with what needs
attention (a source that failed, went quiet, shrank or was cut short, a model that
did not run or whose rates jumped), then the roles, then this section. The roles
get the email's room first: if an email holds more role cards than fit, the later
ones become one line each, and the digest on disk keeps them all. This section
gets what the roles leave, so a high `hidden_max` never pushes a role out; the
email always keeps its heading, every rule's count and where the rest is, and as
many rows as fit, which the section says when it is fewer than the cap.

**Re-assessing needs more than `--dry`.** Every posting a real scan assessed is
recorded as seen, the ones under a threshold included, and a `--dry` run drops
seen postings like any other. So after a real scan, lowering a threshold and
re-running `--dry` finds nothing new. Either run `rolescan unsee URL` for a
posting you know of, or tune against a scratch store: point `output.db_path` at a
new file (and `output.dir` if you like), run `--dry` as often as you want, then
point it back.

**A hidden posting is listed once.** Everything a scan assesses is recorded as
seen, hidden postings included, and is not reported again; a `--dry` run records
nothing, so it lists them every time. The one exception is a yearly programme:
a posting whose title names one ("Summer Internship", "Graduate Programme",
"Off-Cycle Internship", "Insight Week") has the same identity every year, so a
board that opens it again would never be heard from twice. Each scan refreshes
`last_seen` on the postings it finds already seen, so a posting no scan has found
for `output.reopen_programme_days` days (120; `0` turns this off) counts as new:
it is scored and listed again, under "Returning: listed again after N days off
the board", and its row takes a fresh `first_seen`. A posting a board lists on
every scan never comes back, nor does one whose title is not a programme, and
your shortlist, applied and dismissed marks are left as they were: a role you
dismissed last year still returns under that line. Read the section on the day you scan: the
digest is also saved under `digests/`, and `rolescan show` reprints the latest
one. If a rule or term was wrong, fix it in `config.yaml`, then
`rolescan unsee URL` (the URL as the digest prints it) so the next scan reports
that posting again. Deleting `seen.db` re-assesses everything, but it also
clears your shortlist, applied and dismissed marks.
