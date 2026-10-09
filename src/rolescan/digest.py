"""Digest rendering and delivery."""

from __future__ import annotations

import html
import logging
import smtplib
import ssl
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, NamedTuple

from rolescan.config import EmailConfig
from rolescan.models import FitVerdict, Job, ScoredJob, Verdict
from rolescan.netif import interface_ipv4
from rolescan.pipeline import ScanResult, SourceReport
from rolescan.scoring.judges import available_judges
from rolescan.scoring.keyword import SYNTHETIC_PENALTIES
from rolescan.scoring.rules import RULE_ORDER

__all__ = [
    "EmailError",
    "hidden_list_needed",
    "hidden_list_path",
    "next_digest_path",
    "render_hidden_list",
    "render_html",
    "render_markdown",
    "send_email",
    "write_digest",
    "write_hidden_list",
]

log = logging.getLogger(__name__)

_BADGE = {
    Verdict.APPLY: "APPLY",
    Verdict.CONSIDER: "CONSIDER",
    Verdict.SKIP: "SKIP",
    Verdict.BLOCKED: "BLOCKED",
}

# Prose shared by the markdown and HTML renderers, as (text, is_code) pairs.
# Markdown wraps a code fragment in backticks and HTML in a <code> span, so
# the sentence itself is written once and neither renderer has to know the
# other's markup. Nothing here reads the finished markdown back.
_Frags = list[tuple[str, bool]]

_KEYWORD_ONLY_FRAGS: _Frags = [
    (
        "Every posting in this digest was ranked on keyword score alone. "
        "Keyword scores are not on the same scale as the fit scores ",
        False,
    ),
    ("min_report_score", True),
    (
        " was set for, so expect this digest to be much shorter than it "
        "should be — or empty — until the backend works. It is not a "
        "quiet market.",
        False,
    ),
]

_PROBE_ONLY_BODY = (
    "Scoring itself worked, so the fit scores in this digest are real. It is "
    "the liveness probe that failed, not the backend — it is deliberately "
    "short so an unattended run cannot hang on it, and a loaded server can "
    "exceed it."
)

_DISCOVER_FRAGS: _Frags = [
    ("Run ", False),
    ("rolescan discover", True),
    (" to check the slugs.", False),
]

# 2.5.8: the two source alarms beside the quiet one, and what each asks of
# the reader.
_SHRUNK_BODY = (
    "A board rarely loses most of its postings between two runs. These "
    "returned rows and raised nothing, which is how a broken page or a moved "
    "cap looks. Check the source before trusting this digest."
)

_CUT_FRAGS: _Frags = [
    ("The roles past the cut are in no digest. For a Workday board, raise ", False),
    ("max_rows", True),
    (", or narrow it with ", False),
    ("applied_facets", True),
    (" or ", False),
    ("search_text", True),
    (".", False),
]


def _md_frags(frags: _Frags) -> str:
    return "".join(f"`{text}`" if code else text for text, code in frags)


# --- Hidden by your rules ---------------------------------------------------
#
# On 2026-10-06, 44 of 109 scored postings were hidden by `profile.rules` with
# no trace: a rule skip is capped below `min_report_score` and a rule block is
# dropped by `show_blocked: false`, so a mis-read advert or a rule bug that
# skipped a good role was invisible. This section lists them, one line each,
# grouped by the rule that fired, so a wrong skip can be spotted. Both
# renderers build from `_rule_hidden_groups`, so only the markup can differ.

#: A short heading per `FitVerdict.rule`. `hard_bar` also covers the skip an
#: `other` bar gives (a driving licence, a sector background), so the label
#: does not claim every one of them is a nationality or clearance bar.
#: `_TERMS` is not a `decide` rule: it groups postings a configured
#: `hard_blockers` term or `excluded_locations` entry blocked
#: (`ScoredJob.blocker_hits`, where a location reads "location: dubai") when
#: no rule fired, and always comes last.
_TERMS = "hard_blockers"
_RULE_LABELS: dict[str, str] = {
    "hard_bar": "Nationality, clearance or other hard bar",
    "graduation_year": "Graduation year",
    "student_only": "Students only",
    "level": "Level",
    "years": "Years of experience",
    "field": "Field",
    _TERMS: "Your blocking terms (hard_blockers, excluded_locations)",
    "blockers": "Your weighted terms (blockers)",
    "gate": "Just under your keyword gate",
}

_RULE_HIDDEN_LEAD = (
    "Skipped or blocked by one of your rules, so not listed above. One line "
    "each, so a wrong skip can be spotted."
)

_REASON_PREFIXES = ("Skip: ", "Blocked: ")


def _hidden_reason(reason: str) -> str:
    """`reason` without its verdict prefix: the heading already says why."""
    for prefix in _REASON_PREFIXES:
        if reason.startswith(prefix):
            return reason.removeprefix(prefix)
    return reason


def _hidden_group(item: ScoredJob, gate: int) -> tuple[str, str] | None:
    """The group a rule-hidden posting is listed under, and its reason.

    The rule `decide` fired wins over a `hard_blockers` term, so a posting
    both caught is listed once, under the rule: its reason quotes the advert,
    which is what a reader checks a skip against. A judge-mode model block
    has no rule (`decide` never ran) and goes under `hard_bar` with the
    model's reason, ahead of any term it also matched. A term-only posting's
    reason is the term(s) that matched, as configured.

    2.5.7: a prefilter reject `_rule_hidden` listed carries `hidden_as`, which
    decides the group after the branches above: `blockers` quotes the
    configured terms that cost it points (the synthetic agency and location
    penalties are not terms), `gate` gives its score against `gate`. It is
    `_rule_hidden`, not this function, that knows the weights and so whether
    a term really pushed the posting under.
    """
    fit = item.fit
    if fit is not None and fit.rule is not None:
        return fit.rule, _hidden_reason(fit.reason)
    if fit is not None and fit.verdict is Verdict.BLOCKED:
        return "hard_bar", _hidden_reason(fit.reason)
    if item.blocker_hits:
        terms = ", ".join(f'"{t}"' for t in dict.fromkeys(item.blocker_hits))
        return _TERMS, f"blocked by {terms}"
    if item.hidden_as == "blockers":
        costly = [
            t
            for t in dict.fromkeys(item.keyword_penalties)
            if t not in SYNTHETIC_PENALTIES
        ]
        return "blockers", "pushed under the gate by " + ", ".join(
            f'"{t}"' for t in costly
        )
    if item.hidden_as == "gate":
        return "gate", f"scored {item.keyword_score} of {gate}"
    return None


class _HiddenRow(NamedTuple):
    rule: str
    heading: str
    job: Job
    reason: str
    score: int


def _rule_hidden_rows(result: ScanResult) -> list[_HiddenRow]:
    """Every posting the section lists, grouped by rule in `decide`'s order,
    then the `hard_blockers` terms group, then (2.5.7) the weighted `blockers`
    group and the just-under-the-gate group, each group sorted by company.

    A rule name this module has no label for (one added to `decide` without
    updating `_RULE_LABELS`) still renders, under its own name and after the
    known rules, rather than vanishing from the one place it is reported.
    """
    groups: dict[str, list[_HiddenRow]] = {}
    for item in result.rule_hidden:
        if (found := _hidden_group(item, result.gate)) is None:
            continue
        rule, reason = found
        groups.setdefault(rule, []).append(
            _HiddenRow(rule, _RULE_LABELS.get(rule, rule), item.job, reason, item.score)
        )
    rank = {rule: i for i, rule in enumerate(RULE_ORDER)}
    rank[_TERMS] = len(RULE_ORDER) + 1
    rank["blockers"] = len(RULE_ORDER) + 2
    rank["gate"] = len(RULE_ORDER) + 3
    rows: list[_HiddenRow] = []
    for rule in sorted(groups, key=lambda r: (rank.get(r, len(RULE_ORDER)), r)):
        rows += sorted(
            groups[rule],
            key=lambda row: (row.job.company.casefold(), row.job.title.casefold()),
        )
    return rows


@dataclass(frozen=True, slots=True)
class _HiddenGroup:
    heading: str
    total: int
    """How many postings this rule hid."""
    rows: list[tuple[Job, str]]
    """The ones listed: all of them unless the section is capped."""

    @property
    def count(self) -> str:
        """ "4", or "2 of 4 listed" when the cap left some out."""
        if len(self.rows) == self.total:
            return str(self.total)
        return f"{len(self.rows)} of {self.total} listed"


@dataclass(frozen=True, slots=True)
class _HiddenView:
    """What the "Hidden by your rules" section shows (2.6.0), built once and
    read by both renderers and the stats line, so the three cannot disagree.

    `groups` holds every group the rules hid something under, in order, even
    one whose rows were all cut; `total` and `listed` count postings. `file`
    is the name of the file holding the whole list, and is "" unless the list
    was cut and the caller named one. `fitted` is True for the email's view
    when the roles left room for fewer rows than the cap allows (see
    `_fit_hidden`)."""

    groups: list[_HiddenGroup]
    total: int
    listed: int
    file: str = ""
    fitted: bool = False

    @property
    def capped(self) -> bool:
        return self.listed < self.total


def _hidden_view(
    result: ScanResult, hidden_max: int | None = None, hidden_file: str = ""
) -> _HiddenView:
    """The groups of `_rule_hidden_rows`, cut to `hidden_max` postings.

    The cut keeps the highest-scoring postings: a hide with a high score is the
    likeliest wrong one, which is what the section is for. Ties fall to company
    and title, so the choice is the same on every render. `hidden_max=None` is
    no cap, which is what a caller that does not know about the cap gets."""
    rows = _rule_hidden_rows(result)
    keep = set(range(len(rows)))
    if hidden_max is not None and len(rows) > hidden_max:
        order = sorted(
            keep,
            key=lambda i: (
                -rows[i].score,
                rows[i].job.company.casefold(),
                rows[i].job.title.casefold(),
                rows[i].job.url,
            ),
        )
        keep = set(order[: max(hidden_max, 0)])
    totals: dict[str, int] = {}
    listed: dict[str, list[tuple[Job, str]]] = {}
    headings: dict[str, str] = {}
    for i, row in enumerate(rows):
        headings.setdefault(row.rule, row.heading)
        totals[row.rule] = totals.get(row.rule, 0) + 1
        listed.setdefault(row.rule, [])
        if i in keep:
            listed[row.rule].append((row.job, row.reason))
    groups = [_HiddenGroup(headings[r], totals[r], listed[r]) for r in totals]
    capped = len(keep) < len(rows)
    return _HiddenView(groups, len(rows), len(keep), hidden_file if capped else "")


def _unread_heading(result: ScanResult) -> str:
    return f"Unread (no text after {_plural(result.unread_after, 'run')})"


def _unread_lead(result: ScanResult) -> str:
    return (
        f"No description text arrived for these on {result.unread_after} runs, "
        "so they were never judged. Listed once, with the link, and not shown "
        "again."
    )


def _unread_note(item: ScoredJob) -> str:
    """A `hard_blockers` term the title matched, as the terms group words it,
    or "". A location entry is not something the reader can check."""
    terms = [
        t for t in dict.fromkeys(item.blocker_hits) if not t.startswith("location: ")
    ]
    return "blocked by " + ", ".join(f'"{t}"' for t in terms) if terms else ""


def _unread_section(result: ScanResult) -> list[str]:
    """One line each for the postings that went `thin_unread_after` runs
    without text (see `ScanResult.unread`). Placed before "Hidden by your
    rules": nothing has judged these, which is a different thing from a rule
    having hidden them."""
    if not result.unread:
        return []
    lines = [f"## {_unread_heading(result)}", "", _unread_lead(result), ""]
    for item in sorted(
        result.unread,
        key=lambda s: (s.job.company.casefold(), s.job.title.casefold()),
    ):
        job = item.job
        bits = [f"**{job.company}**", f"[{job.title}]({job.url})"]
        if job.location:
            bits.append(job.location)
        line = f"- {' · '.join(bits)}"
        if note := _unread_note(item):
            line += f": {note}"
        lines.append(line)
    lines.append("")
    return lines


def _hidden_counts(view: _HiddenView) -> str:
    """ "Level 200, Years of experience 150": what each rule hid."""
    return ", ".join(f"{g.heading} {g.total}" for g in view.groups)


def _capped_frags(view: _HiddenView) -> _Frags:
    """The sentence that opens a capped "Hidden by your rules" section: every
    rule's count, which postings are listed, and where the rest are."""
    shown = (
        f"Listed below: {_plural(view.listed, 'posting')} with the highest scores."
        if view.listed
        else "None are listed below."
    )
    frags: _Frags = [(f"{view.total} hidden: {_hidden_counts(view)}. {shown}", False)]
    if view.file:
        frags += [(f" All {view.total} are in ", False), (view.file, True)]
        frags.append((", beside this digest.", False))
    elif view.fitted:
        # `output.hidden_max` did not cut this list, so there is no file and
        # none may be named: the Markdown digest holds all of it.
        frags.append((f" All {view.total} are in the digest on disk.", False))
    if view.file or not view.fitted:
        frags += [(" The cap is ", False), ("output.hidden_max", True), (".", False)]
    if view.fitted:
        frags.append(
            (
                " The email lists only what fits, after the roles, under the "
                "size Gmail clips at.",
                False,
            )
        )
    return frags


def _hidden_line_md(job: Job, reason: str) -> str:
    bits = [f"**{job.company}**", f"[{job.title}]({job.url})"]
    if job.location:
        bits.append(job.location)
    return f"- {' · '.join(bits)}: {reason}"


def _hidden_groups_md(view: _HiddenView) -> list[str]:
    lines: list[str] = []
    for group in view.groups:
        if not group.rows:
            continue
        lines += [f"**{group.heading}** ({group.count})", ""]
        lines += [_hidden_line_md(job, reason) for job, reason in group.rows]
        lines.append("")
    return lines


def _rule_hidden_section(view: _HiddenView) -> list[str]:
    if not view.total:
        return []
    lines = ["## Hidden by your rules", "", _RULE_HIDDEN_LEAD, ""]
    if view.capped:
        lines += [_md_frags(_capped_frags(view)), ""]
    return lines + _hidden_groups_md(view)


def render_hidden_list(result: ScanResult) -> str:
    """The whole "Hidden by your rules" list as Markdown, or "" when nothing
    was hidden (2.6.0): what the digest's capped section leaves out, written
    beside it when `output.hidden_max` cut the list."""
    view = _hidden_view(result)
    if not view.total:
        return ""
    stamp = datetime.now(UTC).strftime("%A %d %B %Y")
    lines = [
        f"# Hidden by your rules, {stamp}",
        "",
        "Every posting one of your rules or terms kept out of the digest in this "
        "scan, one line each, so a wrong skip can be spotted.",
        "",
        f"{view.total} hidden: {_hidden_counts(view)}.",
        "",
    ]
    return "\n".join(lines + _hidden_groups_md(view))


def _returning(item: ScoredJob) -> str:
    """What a programme that came back says about itself (2.7.0), or "".

    The same posting was listed in an earlier year; the reader is told it is
    back rather than left to wonder why a role they dealt with returned."""
    if not item.reopened_after_days:
        return ""
    return (
        f"listed again after {_plural(item.reopened_after_days, 'day')} off the board"
    )


def _role(item: ScoredJob) -> list[str]:
    job, fit = item.job, item.fit
    badge = _BADGE[item.verdict]
    lines = [f"### {job.title}", ""]

    meta = [f"**{job.company}**", job.location or "location not stated"]
    if job.posted:
        meta.append(f"posted {job.posted.isoformat()}")
    lines.append(" | ".join(meta))
    lines.append("")
    lines.append(
        f"`{badge}` **{item.score}/100**"
        + (f" · {fit.confidence.value} confidence" if fit else " · keyword only")
    )
    lines.append("")
    if returning := _returning(item):
        lines.append(f"**Returning:** {returning}")
        lines.append("")

    if fit is not None:
        lines.append(fit.reason)
        lines.append("")
        # A configured blocker can override the LLM's own verdict, in which
        # case fit.blockers (what the model itself noticed) may be empty even
        # though this posting is blocked. Show whichever list is non-empty so
        # a blocked entry never renders with no stated reason.
        blocked_by = list(dict.fromkeys([*fit.blockers, *item.blocker_hits]))
        if blocked_by:
            lines.append("**Blocked by:** " + "; ".join(blocked_by))
            lines.append("")
        # No gap advice for a role you cannot be considered for: it reads as
        # an invitation to waste an afternoon on it.
        if not item.is_blocked and fit.keywords_missing:
            lines.append("**Gaps:** " + ", ".join(fit.keywords_missing))
            lines.append("")
    else:
        # Same rule as the LLM branch, for the keyword-only path: a BLOCKED
        # badge must never appear without the term that caused it. A hard
        # blocker does not have to carry a weight, so it need not show up in
        # keyword_penalties below - and if it does not, nothing else here
        # names it.
        if item.blocker_hits:
            blocked_by = list(dict.fromkeys(item.blocker_hits))
            lines.append("**Blocked by:** " + "; ".join(blocked_by))
            lines.append("")
        if item.keyword_penalties:
            lines.append("**Flags:** " + ", ".join(sorted(set(item.keyword_penalties))))
            lines.append("")
        hits = list(dict.fromkeys(item.keyword_hits))[:8]
        if hits:
            lines.append("**Matched:** " + ", ".join(hits))
            lines.append("")

    lines.append(f"[Apply]({job.url})")
    lines.append("")
    return lines


def _shortlist_section(
    shortlist: list[tuple[str, str, str]], config_path: Path | None
) -> list[str]:
    # `rolescan mark` defaults to ./config.yaml, so a command printed without
    # --config only works from the project directory. Under launchd the config
    # is at an absolute path and the reader is on a phone: they copy the line
    # into whatever shell they have open and get "No config at config.yaml".
    flag = f" --config {config_path}" if config_path is not None else ""
    lines = [
        "",
        "## Shortlist",
        "",
        "Still open, not yet applied to or dismissed.",
        "",
    ]
    for url, company, role in shortlist:
        company = company.strip()
        role = role.strip()
        if company and role:
            headline = f"**{company}** — {role}"
        elif company or role:
            headline = f"**{company or role}**"
        else:
            # A `mark shortlist` on a url that was never scanned leaves
            # company/title blank. Never emit a bullet with nothing a human
            # can act on: fall back to the one thing we do have.
            headline = url
        lines.append(f"- {headline}")
        lines.append(f"  {url}")
        lines.append(
            f"  `rolescan mark {url} applied{flag}` · "
            f"`rolescan mark {url} dismissed{flag}`"
        )
    return lines


def render_markdown(
    result: ScanResult,
    *,
    title: str = "Job scan",
    shortlist: list[tuple[str, str, str]] | None = None,
    config_path: Path | None = None,
    hidden_max: int | None = None,
    hidden_file: str = "",
) -> str:
    """The digest as Markdown.

    Order (2.6.0): alarms, roles, unread, hidden, shortlist. The alarms are
    first because they are what a reader must not miss, and a long list below
    them can no longer push them out of sight.

    `hidden_max` caps the "Hidden by your rules" section at that many lines,
    counting every rule, and `hidden_file` names the file holding the whole
    list so the digest can say where it is. None is no cap, for a caller that
    does not know about the cap."""
    stamp = datetime.now(UTC).strftime("%A %d %B %Y")
    out: list[str] = [f"# {title}, {stamp}", ""]
    view = _hidden_view(result, hidden_max, hidden_file)

    if result.dry_run:
        out += ["*Dry run: nothing was marked as seen.*", ""]

    out += _failures(result)

    if not result.reportable:
        out += [
            "Nothing new worth your time today.",
            "",
            _stats(result, view),
            "",
        ]
        out += _unread_section(result)
        out += _rule_hidden_section(view)
        if shortlist:
            out += _shortlist_section(shortlist, config_path)
        return "\n".join(out)

    live = [s for s in result.reportable if not s.is_blocked]
    blocked = [s for s in result.reportable if s.is_blocked]

    out.append(
        f"**{len(live)} worth a look**"
        + (f", {len(blocked)} blocked" if blocked else "")
        + f". {_stats(result, view)}"
    )
    out.append("")

    if live:
        out += ["## Worth a look", ""]
        for item in live:
            out += _role(item)

    if blocked:
        out += [
            "## Blocked",
            "",
            "Structurally closed to you. Listed so you know the market moved, "
            "not as options.",
            "",
        ]
        for item in blocked:
            out += _role(item)

    out += _unread_section(result)

    out += _rule_hidden_section(view)
    if shortlist:
        out += _shortlist_section(shortlist, config_path)
    return "\n".join(out)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


#: The baseline sizes the Model health line can name (3 to 5 earlier runs).
_RUN_WORDS = {3: "three", 4: "four", 5: "five"}


def _run_outcome_note(result: ScanResult) -> str:
    """A short, accurate note when no source ran cleanly, or "".

    Zero reportable roles because nothing matched looks identical, in the
    digest, to zero because nothing ran. This is what tells the reader
    which one they are looking at — without conflating "a source errored"
    (a scan failure) with "a source declined to run because it is not
    configured" (a config gap, not a failure, and the normal state for
    e.g. Adzuna with no API key). Getting this wrong sends someone to debug
    a working scraper at 06:30.
    """
    reports = result.reports
    if not reports or any(r.ok for r in reports):
        return ""
    errored = [r for r in reports if r.error and not r.skipped]
    skipped = [r for r in reports if r.skipped]
    if errored and not skipped:
        return (
            " Every source failed this run. This is a scan failure, not an "
            "empty market."
        )
    if skipped and not errored:
        return (
            " No configured source was able to run this time — nothing "
            "errored, they simply declined to run. Check credentials or "
            "config for the skipped source(s)."
        )
    # A mix: true of both halves without calling the whole run "failed".
    return (
        f" Nothing ran cleanly this run: {len(errored)} source(s) errored, "
        f"{len(skipped)} declined to run (not configured). See below."
    )


def _minutes(n: float) -> str:
    """A budget in minutes as a reader says it: "10 minutes", "7.5 minutes",
    "1 minute"."""
    shown = str(int(n)) if n == int(n) else str(n)
    return f"{shown} minute" if n == 1 else f"{shown} minutes"


#: How the stats line names each reason a posting was deferred, in order.
#: `llm_time` is not here: it has a clause of its own (see `_stats`), because
#: it names the budget that was spent.
_DEFERRED_LABELS = {
    "llm_ceiling": "over the LLM budget",
    "llm_breaker": "after the LLM stopped answering",
    "digest_cap": "over the digest cap",
    "thin": "without text yet",
}


def _stats(result: ScanResult, view: _HiddenView) -> str:
    live = sum(1 for r in result.reports if r.ok)
    bits = [
        f"Scanned {_plural(result.unique, 'unique posting')} from "
        f"{_plural(live, 'source')}",
        f"{result.already_seen} already seen",
        f"{result.prefiltered} filtered before scoring",
    ]
    if result.stale:
        # Only when it happened. ATS boards serve evergreen requisitions, so
        # this number is how the reader learns their new sources carry old
        # stock rather than wondering why a board of 228 yielded nothing.
        bits.insert(1, f"{result.stale} too old")
    if result.llm_scored:
        # Disjoint figures: scored by a call this run, and served from the
        # cache. A run of cache hits alone still ran (2.5.8).
        bits.append(
            f"{result.llm_scored - result.llm_cached} scored, "
            f"{result.llm_cached} from cache" + _model_note(result)
        )
    if result.below_min_report_score:
        # Only when it happened. These scored and fell under the final gate,
        # so they are in no list: the number is what tells "nothing matched"
        # from "everything matched weakly".
        bits.append(f"{result.below_min_report_score} below min_report_score")
    if result.facts_cache_skipped:
        bits.append("facts cache skipped: model identity unknown")
    if result.hidden_blocked:
        # Only when it happened: a permanent "0 blocked and hidden" on every
        # digest would train the reader to skip the line that matters. These
        # postings scored well enough to be here and were removed anyway, and
        # they are recorded as seen once this digest is written (except a
        # deferred one, an unjudged one when the backend broke, and anything
        # on a dry run, which are not recorded and come round again), so for
        # the rest this is the reader's only chance to notice a blocker term
        # that is matching the wrong thing.
        bits.append(
            f"{result.hidden_blocked} blocked and hidden (output.show_blocked is false)"
        )
    if timed_out := sum(1 for s in result.deferred if s.deferred == "llm_time"):
        # The model's time budget was spent (2.7.0). Its own clause, ahead of
        # the general one, because the number a reader needs is the budget:
        # the call-count wording ("over the LLM budget") would send them to
        # `max_calls_per_run`, which was not what stopped the run.
        bits.append(
            f"model time budget of {_minutes(result.llm_max_minutes)} spent; "
            f"{_plural(timed_out, 'posting')} deferred to the next run"
        )
    if others := [s for s in result.deferred if s.deferred != "llm_time"]:
        reasons = Counter(s.deferred for s in others)
        parts = [
            f"{reasons[key]} {label}"
            for key, label in _DEFERRED_LABELS.items()
            if reasons.get(key)
        ]
        # A marker with no label yet is named as it is, never dropped: an
        # empty "()" read as nothing held back (2.5.8).
        parts += [
            f"{n} {key}" for key, n in reasons.items() if key not in _DEFERRED_LABELS
        ]
        bits.append(f"{len(others)} deferred to the next run ({', '.join(parts)})")
    if result.unread:
        bits.append(f"{len(result.unread)} listed as unread")
    if view.total:
        # Only when it happened, for the same reason. Counted separately from
        # the clause above: that one is postings `show_blocked` removed, this
        # one is postings any rule or `hard_blockers` term removed by any
        # route, so a block hidden by `show_blocked` is in both - hence
        # "listed below", so the two numbers do not read as two postings. It
        # is read from the same view the section is, so the number and the
        # list can never disagree. A capped section says how many it lists
        # and where the whole list is (2.6.0).
        where = "listed below"
        if view.capped:
            where = f"{view.listed} listed below"
            if view.file:
                where += f", all {view.total} in {view.file}"
        bits.append(f"{view.total} hidden by your rules ({where})")
    return ". ".join(bits) + "." + _run_outcome_note(result)


def _model_note(result: ScanResult) -> str:
    """ " by <model> (<digest>)", or "" when the run named no model (2.5.8).

    A verdict is only as reproducible as the model behind it, and a tag can
    be re-pulled under the same name: the digest says which weights ran."""
    if not result.llm_model:
        return ""
    digest = f" ({result.llm_model_digest})" if result.llm_model_digest else ""
    return f" by {result.llm_model}{digest}"


def _llm_ran(result: ScanResult) -> bool:
    """Whether any posting in this digest actually carries an LLM score.

    `llm_scored` counts postings that got a verdict, by a call or from the
    cache. `llm_calls` counts calls attempted, so five failed calls read as
    scoring that ran (2.5.8)."""
    return bool(result.llm_scored)


def _not_run_head(result: ScanResult) -> tuple[str, str]:
    """(the lead, the rest) of the "LLM scoring did not run" note.

    A model window below `llm.num_ctx` is one sentence with the fix in it
    (2.5.8): the server is up and the model is pulled, so it must not read
    like an outage. Every other reason follows "did not run at all"."""
    if result.llm_window_stop:
        return f"LLM scoring did not run: {result.llm_unusable}.", ""
    return "LLM scoring did not run at all.", f"{result.llm_unusable}."


def _llm_error_hint(result: ScanResult) -> _Frags:
    """One line naming the likely cause, for the backend that was configured.

    "A 401 here means ANTHROPIC_API_KEY is missing" is the right sentence for
    the hosted backend and exactly the wrong one for `ollama returned HTTP
    500`: it sends someone to look at a credential that backend has never
    had. This is the place the end user actually reads, so the distinction
    has to be real here, not only in the CLI.

    Returned as fragments rather than a finished string so the HTML digest can
    set the config keys in a code span and escape the backend name, without
    either renderer parsing the other's markup.
    """
    judge = available_judges().get(result.llm_backend)
    if judge is None:
        return []
    if judge.needs_api_key:
        env = judge.api_key_env or "the backend's API key"
        return [
            ("A 401 here means ", False),
            # Deliberately not a code span: the markdown has never set it in
            # one, and a test asserts the bare name appears.
            (env, False),
            (" is missing, revoked, or from another organisation.", False),
        ]
    return [
        ("The ", False),
        (result.llm_backend, True),
        (
            " backend needs no API key, so a credential is not the cause. "
            "Check the server is still up and still holding ",
            False,
        ),
        ("llm.model", True),
        (", and that ", False),
        ("llm.timeout", True),
        (" is long enough for it.", False),
    ]


def _model_health_text(result: ScanResult) -> str:
    """The "Model health" line without its heading (2.6.0): each rate this run
    that is over twice the median of the last three to five runs (as many as
    there were), with that median.

    A reader who sees it has one thing to do, which is to look at the model:
    the quote guard and the resolvers correct what the model says, and a rate
    that doubles means the model, its prompt or a resolver changed."""
    flags = result.model_health
    rates = ", ".join(
        f"{f.label} {round(100 * f.affected / f.postings)}% "
        f"(median {round(100 * f.median)}%)"
        for f in flags
    )
    runs = flags[0].runs
    return (
        f"More than twice the median of the last {_RUN_WORDS.get(runs, runs)} "
        f"runs, over "
        f"{_plural(flags[0].postings, 'posting')}: {rates}. The model, its "
        "prompt or a resolver may have changed; check before trusting this digest."
    )


def _failures(result: ScanResult) -> list[str]:
    """The alarm block, which opens the digest (2.6.0): sources that failed,
    went quiet, shrank or were cut short, a model that did not run, and the
    notes. Empty when nothing needs attention."""
    failed = result.failed_sources
    skipped = result.skipped_sources
    if (
        not failed
        and not skipped
        and not result.quiet_sources
        and not _source_alarms(result)
        and not result.llm_errors
        and not result.model_health
        and not result.llm_unusable
        and not result.enricher_unusable
    ):
        return []
    lines = ["## Needs attention", ""]
    if result.llm_unusable and not _llm_ran(result):
        # The configured judge never ran at all, so there are no scoring errors
        # to report and the digest would otherwise look like a normal quiet
        # day. It is not one: every posting fell back to a keyword score, and
        # keyword scores are not calibrated against min_report_score, so this
        # digest is close to empty by construction rather than by market.
        lead, rest = _not_run_head(result)
        lines += [
            f"**{lead}**" + (f" {rest}" if rest else ""),
            "",
            _md_frags(_KEYWORD_ONLY_FRAGS),
            "",
        ]
    elif result.llm_unusable:
        # The preflight said the backend was unusable and then scoring worked.
        # Claiming nothing was scored, above postings carrying fit scores and
        # confidence levels, makes the digest contradict itself and teaches
        # the reader to disbelieve the warning on the day it is true. The
        # probe is a few seconds by design, and a server part-way through
        # loading a large model can exceed it.
        lines += [
            f"**The pre-scan backend check failed, but scoring ran anyway.** "
            f"{result.llm_unusable}.",
            "",
            _PROBE_ONLY_BODY,
            "",
        ]
    if result.enricher_unusable:
        # Deliberately separate from the two branches above: the judge itself
        # is fine, only the (optional) extra step an enricher adds is not, so
        # this must never read like "scoring did not run" or "the backend
        # check failed" - those are a different failure entirely.
        lines += [
            f"**{result.enricher_unusable}.** Postings are scored normally, "
            "just without that extra step.",
            "",
        ]
    if result.llm_errors:
        # Every score fell back to keywords. Without this the digest is
        # indistinguishable from a deliberate --no-llm run.
        lines += [
            f"**LLM scoring failed for {result.llm_errors} posting(s)** "
            "— those roles are ranked on keyword score alone.",
            "",
            f"`{result.llm_error_detail}`" if result.llm_error_detail else "",
            "",
        ]
        # Guarded because the hint is empty for a backend nothing registered,
        # and an unconditional append then emits a stray blank line where the
        # advice should be. run_scan always sets llm_backend, so this is
        # unreachable from a real scan; a ScanResult built anywhere else is
        # not, and losing the advice silently is the failure this section
        # exists to prevent.
        if hint := _llm_error_hint(result):
            lines += [_md_frags(hint), ""]
    if result.model_health:
        lines += [f"**Model health**: {_model_health_text(result)}", ""]
    if result.quiet_sources:
        # Above the failures on purpose. A source that errors says so; a source
        # that quietly returns nothing is the defect this project keeps
        # producing, and until now the only sign was a thinner digest.
        lines += ["**Sources that went quiet**", ""]
        lines += [
            f"- **{label}** returned nothing, having returned up to {n} recently"
            for label, n in result.quiet_sources
        ]
        lines += [
            "",
            "That is not a quiet market: these worked before and returned "
            "nothing now. Check the source before trusting this digest.",
            "",
        ]
    lines += _shrunk_and_cut_md(result)
    if failed:
        lines += ["**Sources that failed this run**", ""]
        lines += [f"- `{r.kind}/{r.slug}` {r.error}" for r in failed]
        lines += ["", _md_frags(_DISCOVER_FRAGS), ""]
    if skipped:
        # Called out separately because a skipped source contributed nothing
        # and is easy to mistake for one that found nothing.
        lines += ["**Sources skipped (not searched)**", ""]
        lines += [f"- `{r.kind}/{r.slug}` {r.error}" for r in skipped]
        lines += [""]
    lines += _notes_md(result)
    return [*lines, "---", ""]


def _source_alarms(result: ScanResult) -> bool:
    """Whether a 2.5.8 source section is due. Each is shown even when it is
    the only thing to report: the rule the quiet alarm set in 2.5.7, when the
    Markdown digest dropped it whenever it was the only problem."""
    return bool(result.shrunk_sources or result.truncated_sources or result.notes)


def _shrunk_line(n: int, median: float) -> str:
    return f"returned {n}, under 30% of its recent median of {median:g}"


def _cut_line(read: int, total: int | None, why: str) -> str:
    """What follows a cut-short source's label: "(500 of 2000 read): <why>",
    or "(60 read): <why>" when the board stated no total."""
    of = f" of {total}" if total is not None else ""
    return f"({read}{of} read): {why}"


def _shrunk_and_cut_md(result: ScanResult) -> list[str]:
    """The "Sources that shrank" and "Sources that were cut short" sections
    (2.5.8), straight after the quiet alarm: the same silent defect, caught
    while the source still returns rows."""
    lines: list[str] = []
    if result.shrunk_sources:
        lines += ["**Sources that shrank**", ""]
        lines += [
            f"- **{label}** {_shrunk_line(n, median)}"
            for label, n, median in result.shrunk_sources
        ]
        lines += ["", _SHRUNK_BODY, ""]
    if result.truncated_sources:
        lines += ["**Sources that were cut short**", ""]
        lines += [
            f"- **{label}** {_cut_line(n, total, why)}"
            for label, n, total, why in result.truncated_sources
        ]
        lines += ["", _md_frags(_CUT_FRAGS), ""]
    return lines


def _notes_md(result: ScanResult) -> list[str]:
    """The "Notes" section (2.5.8): last, and apart from the failures,
    because a posting skipped as unreadable or a backlog waiting is not a
    broken source."""
    if not result.notes:
        return []
    lines = [f"- **{label}**: {note}" for label, note in result.notes]
    return ["**Notes**", "", *lines, ""]


# --- HTML -----------------------------------------------------------------
#
# The HTML alternative is built from the same ScanResult `render_markdown`
# reads, not from the finished markdown. Escaping the markdown into a <pre>
# shipped the SOURCE as the HTML part: `[Apply](https://...)` arrived as
# exactly those characters, so the one link the reader needs was not
# clickable and every `**bold**` was asterisks. Nothing below parses markdown.
#
# Constraints the mail clients impose, all of which this obeys: inline styles
# only (stylesheets and <style> blocks are stripped), no web fonts, no
# JavaScript, no flexbox or grid, one column, and tap targets big enough for
# a thumb.


def _esc(text: str) -> str:
    """Escape for a text node AND for an attribute value.

    `quote=True` is the whole point: every string here is scraped from a
    third-party page, and without it a company name containing a double quote
    closes the style attribute it sits next to and opens whatever follows.
    """
    return html.escape(text, quote=True)


_SAFE_SCHEMES = ("http://", "https://")


def _href(url: str) -> str:
    """An escaped href for `url`, or "" when it must not become a link.

    An allowlist, not a `javascript:` denylist. Job urls come from scraped
    pages and from whatever `rolescan mark` was handed, so a hostile scheme
    is an input this function will see rather than a hypothetical, and a
    denylist loses to the first encoding trick. Anything that is not
    literally http:// or https:// is rendered as inert text instead.
    """
    candidate = url.strip()
    if candidate.casefold().startswith(_SAFE_SCHEMES):
        return _esc(candidate)
    return ""


def _code(token: str) -> str:
    return f'<code style="{_CODE}">{_esc(token)}</code>'


def _html_frags(frags: _Frags) -> str:
    return "".join(_code(text) if code else _esc(text) for text, code in frags)


_BODY = (
    "margin:0;padding:0;background:#f1f2f4;color:#16181d;font-family:"
    "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,"
    "sans-serif;font-size:16px;line-height:1.5;-webkit-text-size-adjust:100%"
)
_SHELL = "max-width:620px;width:100%"
_CELL = "padding:0;text-align:left;word-break:break-word"
_H1 = "margin:0 0 14px;font-size:21px;line-height:1.3;font-weight:700"
_H2 = (
    "margin:26px 0 10px;font-size:13px;font-weight:700;letter-spacing:.1em;"
    "text-transform:uppercase;color:#5b626c"
)
_LEAD = "margin:0 0 14px;color:#4a5057;font-size:15px"
_STATS = "margin:0 0 18px;color:#4a5057;font-size:14px"
_DRY = "margin:0 0 14px;color:#8a5a00;font-size:14px;font-style:italic"
_CARD = (
    "background:#ffffff;border:1px solid #e1e3e8;border-radius:10px;"
    "padding:14px 16px;margin:0 0 12px;"
)
_TITLE = "margin:0 0 4px;font-size:18px;line-height:1.3;font-weight:700"
_TITLE_LINK = "color:#0b4f9e;text-decoration:none"
_COMPANY = "margin:0;font-size:15px;font-weight:600;color:#23262c"
_META = "margin:2px 0 10px;font-size:13px;color:#6b7280"
_BADGE_ROW = "margin:0 0 10px"
_BADGE_CHIP = (
    "display:inline-block;padding:3px 9px;border-radius:999px;font-size:11px;"
    "font-weight:700;letter-spacing:.08em;vertical-align:middle;"
)
_SCORE = "margin-left:9px;font-size:17px;font-weight:700;vertical-align:middle"
_SCORE_DEN = "font-size:13px;color:#6b7280;vertical-align:middle"
_CONF = "margin-left:9px;font-size:13px;color:#6b7280;vertical-align:middle"
_REASON = "margin:0 0 10px;font-size:15px;color:#23262c"
_FLAG = "margin:0 0 8px;font-size:13px;color:#4a5057"
_SUB = "margin:10px 0 0;padding-top:10px;border-top:1px solid #eceef1"
_SUB_LINE = "margin:0 0 6px;font-size:13px;color:#5b626c"
_SUB_LABEL = (
    "margin:0 0 4px;font-size:11px;font-weight:700;color:#8a919b;"
    "letter-spacing:.08em;text-transform:uppercase"
)
_UL = "margin:0 0 6px;padding-left:20px"
_LI = "margin:0 0 4px;font-size:13px;color:#5b626c"
_BTN_ROW = "margin:14px 0 0"
_BTN = (
    "display:inline-block;padding:12px 22px;background:#0b4f9e;color:#ffffff;"
    "text-decoration:none;border-radius:8px;font-size:15px;font-weight:600"
)
_BTN_MUTED = (
    "display:inline-block;padding:12px 22px;background:#ffffff;color:#4a5057;"
    "text-decoration:none;border:1px solid #c9ced6;border-radius:8px;"
    "font-size:15px;font-weight:600"
)
_DEAD_LINK = "margin:14px 0 0;font-size:13px;color:#8a1d1d"
_RULE = "border:0;border-top:1px solid #dcdfe4;margin:26px 0 18px"
_NOTE = (
    "background:#ffffff;border:1px solid #e1e3e8;border-radius:10px;"
    "padding:12px 14px;margin:0 0 12px"
)
_NOTE_HEAD = "margin:0 0 6px;font-size:14px;font-weight:700"
_NOTE_BODY = "margin:0 0 6px;font-size:13px;color:#4a5057"
_CODE = (
    "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
    "font-size:12px;background:#eceef1;border-radius:4px;padding:1px 5px;"
    "word-break:break-all"
)
_LINK = "color:#0b4f9e"
_SL_HEAD = "margin:0 0 4px;font-size:15px;font-weight:700"
_SL_URL = "margin:0 0 8px;font-size:13px"
_SL_CMD = "margin:0 0 4px;font-size:12px;color:#5b626c"

_VERDICT_COLOURS = {
    # badge background, badge text, card rule. The four have to be
    # distinguishable at a glance and without reading the word: apply is a
    # solid green, consider a soft amber, skip a flat grey and blocked a
    # solid red. Text colours are chosen against their own background rather
    # than inherited, because a mail client may impose its own body colour.
    Verdict.APPLY: ("#0a6c39", "#ffffff", "#0a6c39"),
    Verdict.CONSIDER: ("#f6e4bb", "#6b4900", "#c98a12"),
    Verdict.SKIP: ("#e6e8ec", "#4a5057", "#aeb5bf"),
    Verdict.BLOCKED: ("#7d1d1d", "#ffffff", "#7d1d1d"),
}


def _meta_line(job: Job) -> str:
    bits = [job.location or "location not stated"]
    if job.posted:
        bits.append(f"posted {job.posted.isoformat()}")
    return " · ".join(bits)


def _title_html(job: Job) -> str:
    title = _esc(job.title)
    if href := _href(job.url):
        return f'<a href="{href}" style="{_TITLE_LINK}">{title}</a>'
    return title


def _badges_html(item: ScoredJob) -> str:
    background, foreground, _ = _VERDICT_COLOURS[item.verdict]
    note = f"{item.fit.confidence.value} confidence" if item.fit else "keyword only"
    return (
        f'<span style="{_BADGE_CHIP}background:{background};color:{foreground}">'
        f"{_BADGE[item.verdict]}</span>"
        f'<span style="{_SCORE}">{item.score}</span>'
        f'<span style="{_SCORE_DEN}">/100</span>'
        f'<span style="{_CONF}">{_esc(note)}</span>'
    )


def _tags_html(label: str, values: list[str], sep: str = ", ") -> str:
    return (
        f'<div style="{_FLAG}"><strong>{_esc(label)}:</strong> '
        f"{_esc(sep.join(values))}</div>"
    )


def _fit_html(item: ScoredJob, fit: FitVerdict) -> list[str]:
    out = [f'<div style="{_REASON}">{_esc(fit.reason)}</div>']
    # Same merge as the markdown renderer: a configured blocker can override
    # the model's own verdict, so fit.blockers may be empty on a blocked role.
    blocked_by = list(dict.fromkeys([*fit.blockers, *item.blocker_hits]))
    if blocked_by:
        out.append(_tags_html("Blocked by", blocked_by, sep="; "))
    if item.is_blocked:
        return out
    sub: list[str] = []
    if fit.keywords_missing:
        sub.append(
            f'<div style="{_SUB_LINE}">Gaps: '
            f"{_esc(', '.join(fit.keywords_missing))}</div>"
        )
    out.append(f'<div style="{_SUB}">{"".join(sub)}</div>')
    return out


def _keyword_html(item: ScoredJob) -> list[str]:
    out: list[str] = []
    if item.blocker_hits:
        out.append(
            _tags_html("Blocked by", list(dict.fromkeys(item.blocker_hits)), sep="; ")
        )
    if item.keyword_penalties:
        out.append(_tags_html("Flags", sorted(set(item.keyword_penalties))))
    if hits := list(dict.fromkeys(item.keyword_hits))[:8]:
        out.append(_tags_html("Matched", hits))
    return out


def _apply_html(item: ScoredJob) -> str:
    """The second link, because a linked title is easy to miss on a phone."""
    href = _href(item.job.url)
    if not href:
        return (
            f'<div style="{_DEAD_LINK}">No link: the url on this posting is '
            f"not http(s). {_esc(item.job.url)}</div>"
        )
    # A blocked role is listed so the reader knows the market moved, not as an
    # option.
    # A button saying "Apply" invites exactly the wasted afternoon the block
    # exists to prevent, so the link stays and the invitation does not.
    label = "View posting" if item.is_blocked else "Apply"
    style = _BTN_MUTED if item.is_blocked else _BTN
    return f'<div style="{_BTN_ROW}"><a href="{href}" style="{style}">{label}</a></div>'


def _role_html(item: ScoredJob) -> str:
    """One role, ordered the way it is read: title, who, badge, why, how."""
    job = item.job
    *_, rule = _VERDICT_COLOURS[item.verdict]
    out = [
        f'<div style="{_CARD}border-left:4px solid {rule}">',
        f'<div style="{_TITLE}">{_title_html(job)}</div>',
        f'<div style="{_COMPANY}">{_esc(job.company)}</div>',
        f'<div style="{_META}">{_esc(_meta_line(job))}</div>',
        f'<div style="{_BADGE_ROW}">{_badges_html(item)}</div>',
    ]
    if returning := _returning(item):
        out.append(_tags_html("Returning", [returning]))
    if (fit := item.fit) is not None:
        out += _fit_html(item, fit)
    else:
        out += _keyword_html(item)
    out.append(_apply_html(item))
    out.append("</div>")
    return "".join(out)


def _note_html(heading: str, bodies: list[str]) -> str:
    """`heading` is plain text and escaped here; `bodies` are already HTML."""
    inner = "".join(f'<div style="{_NOTE_BODY}">{b}</div>' for b in bodies)
    return (
        f'<div style="{_NOTE}"><div style="{_NOTE_HEAD}">{_esc(heading)}</div>'
        f"{inner}</div>"
    )


def _sources_html(reports: list[SourceReport]) -> str:
    bullets = "".join(
        f'<li style="{_LI}">{_code(f"{r.kind}/{r.slug}")} {_esc(r.error)}</li>'
        for r in reports
    )
    return f'<ul style="{_UL}">{bullets}</ul>'


def _llm_notes_html(result: ScanResult) -> list[str]:
    """The HTML counterpart of the LLM half of `_failures`.

    The branches are the same four and must stay that way: scoring never ran,
    the probe failed but scoring ran anyway, the (optional) enricher could not
    be used, and some calls failed. The prose itself is shared, so only these
    conditions can drift.
    """
    out: list[str] = []
    if result.llm_unusable and not _llm_ran(result):
        out.append(
            _note_html(
                " ".join(filter(None, _not_run_head(result))),
                [_html_frags(_KEYWORD_ONLY_FRAGS)],
            )
        )
    elif result.llm_unusable:
        out.append(
            _note_html(
                "The pre-scan backend check failed, but scoring ran anyway. "
                f"{result.llm_unusable}.",
                [_esc(_PROBE_ONLY_BODY)],
            )
        )
    if result.enricher_unusable:
        # Separate from the two branches above on purpose: the judge itself is
        # fine here, only the (optional) extra step an enricher adds is not.
        out.append(
            _note_html(
                f"{result.enricher_unusable}.",
                ["Postings are scored normally, just without that extra step."],
            )
        )
    if result.llm_errors:
        bodies = []
        if result.llm_error_detail:
            bodies.append(_code(result.llm_error_detail))
        if hint := _llm_error_hint(result):
            bodies.append(_html_frags(hint))
        out.append(
            _note_html(
                f"LLM scoring failed for {result.llm_errors} posting(s) — "
                "those roles are ranked on keyword score alone.",
                bodies,
            )
        )
    if result.model_health:
        out.append(_note_html("Model health", [_esc(_model_health_text(result))]))
    return out


def _failures_html(result: ScanResult) -> list[str]:
    failed = result.failed_sources
    skipped = result.skipped_sources
    if (
        not failed
        and not skipped
        and not result.quiet_sources
        and not _source_alarms(result)
        and not result.llm_errors
        and not result.model_health
        and not result.llm_unusable
        and not result.enricher_unusable
    ):
        return []
    out = [f'<h2 style="{_H2}">Needs attention</h2>', *_llm_notes_html(result)]
    if result.quiet_sources:
        out.append(
            _note_html(
                "Sources that went quiet",
                [
                    "<ul>"
                    + "".join(
                        f"<li><strong>{_esc(label)}</strong> returned nothing, "
                        f"having returned up to {n} recently</li>"
                        for label, n in result.quiet_sources
                    )
                    + "</ul>",
                    "<p>That is not a quiet market: these worked before and "
                    "returned nothing now. Check the source before trusting "
                    "this digest.</p>",
                ],
            )
        )
    out += _shrunk_and_cut_html(result)
    if failed:
        out.append(
            _note_html(
                "Sources that failed this run",
                [_sources_html(failed), _html_frags(_DISCOVER_FRAGS)],
            )
        )
    if skipped:
        out.append(
            _note_html("Sources skipped (not searched)", [_sources_html(skipped)])
        )
    out += _notes_html(result)
    return [*out, f'<hr style="{_RULE}">']


def _labelled_list_html(rows: list[tuple[str, str]], sep: str = " ") -> str:
    """`<ul>` of "<strong>label</strong><sep>text", every part escaped."""
    items = "".join(
        f"<li><strong>{_esc(label)}</strong>{_esc(sep)}{_esc(text)}</li>"
        for label, text in rows
    )
    return f"<ul>{items}</ul>"


def _shrunk_and_cut_html(result: ScanResult) -> list[str]:
    """The HTML counterpart of `_shrunk_and_cut_md`."""
    out: list[str] = []
    if result.shrunk_sources:
        rows = [(lb, _shrunk_line(n, m)) for lb, n, m in result.shrunk_sources]
        out.append(
            _note_html(
                "Sources that shrank",
                [_labelled_list_html(rows), f"<p>{_esc(_SHRUNK_BODY)}</p>"],
            )
        )
    if result.truncated_sources:
        rows = [
            (lb, _cut_line(n, t, why)) for lb, n, t, why in result.truncated_sources
        ]
        out.append(
            _note_html(
                "Sources that were cut short",
                [_labelled_list_html(rows), f"<p>{_html_frags(_CUT_FRAGS)}</p>"],
            )
        )
    return out


def _notes_html(result: ScanResult) -> list[str]:
    """The HTML counterpart of `_notes_md`."""
    if not result.notes:
        return []
    return [_note_html("Notes", [_labelled_list_html(result.notes, sep=": ")])]


def _shortlist_html(
    shortlist: list[tuple[str, str, str]], config_path: Path | None
) -> list[str]:
    flag = f" --config {config_path}" if config_path is not None else ""
    out = [
        f'<h2 style="{_H2}">Shortlist</h2>',
        f'<div style="{_LEAD}">Still open, not yet applied to or dismissed.</div>',
    ]
    for url, company, role in shortlist:
        company, role = company.strip(), role.strip()
        # A `mark shortlist` on a url that was never scanned leaves company
        # and role blank: fall back to the one thing we do have.
        headline = f"{company} — {role}" if company and role else company or role
        href = _href(url)
        link = (
            f'<a href="{href}" style="{_LINK}">{_esc(url)}</a>' if href else _esc(url)
        )
        out.append(
            f'<div style="{_CARD}">'
            f'<div style="{_SL_HEAD}">{_esc(headline or url)}</div>'
            f'<div style="{_SL_URL}">{link}</div>'
            f'<div style="{_SL_CMD}">'
            f"{_code(f'rolescan mark {url} applied{flag}')}</div>"
            f'<div style="{_SL_CMD}">'
            f"{_code(f'rolescan mark {url} dismissed{flag}')}</div>"
            "</div>"
        )
    return out


def _job_line_html(job: Job) -> str:
    """Company, linked title and location, joined by dots."""
    title = _esc(job.title)
    if href := _href(job.url):
        title = f'<a href="{href}" style="{_LINK}">{title}</a>'
    bits = [f"<strong>{_esc(job.company)}</strong>", title]
    if job.location:
        bits.append(_esc(job.location))
    return " · ".join(bits)


def _rule_hidden_line_html(job: Job, reason: str) -> str:
    return f"{_job_line_html(job)}: {_esc(reason)}"


def _unread_html(result: ScanResult) -> list[str]:
    """The HTML counterpart of `_unread_section`."""
    if not result.unread:
        return []
    items = []
    for item in sorted(
        result.unread,
        key=lambda s: (s.job.company.casefold(), s.job.title.casefold()),
    ):
        line = _job_line_html(item.job)
        if note := _unread_note(item):
            line += f": {_esc(note)}"
        items.append(f'<li style="{_LI}">{line}</li>')
    return [
        f'<h2 style="{_H2}">{_esc(_unread_heading(result))}</h2>',
        f'<div style="{_LEAD}">{_esc(_unread_lead(result))}</div>',
        f'<ul style="{_UL}">{"".join(items)}</ul>',
    ]


def _rule_hidden_html(view: _HiddenView) -> list[str]:
    """The HTML counterpart of `_rule_hidden_section`."""
    if not view.total:
        return []
    out = [
        f'<h2 style="{_H2}">Hidden by your rules</h2>',
        f'<div style="{_LEAD}">{_esc(_RULE_HIDDEN_LEAD)}</div>',
    ]
    if view.capped:
        out.append(f'<div style="{_LEAD}">{_html_frags(_capped_frags(view))}</div>')
    for group in view.groups:
        if not group.rows:
            continue
        items = "".join(
            f'<li style="{_LI}">{_rule_hidden_line_html(job, reason)}</li>'
            for job, reason in group.rows
        )
        out += [
            f'<div style="{_SUB_LABEL}">{_esc(group.heading)} ({group.count})</div>',
            f'<ul style="{_UL}">{items}</ul>',
        ]
    return out


#: What the HTML part may weigh, in bytes (2.6.0). Gmail clips a message past
#: about 102 KB and hides the rest behind a link, so the digest stays well
#: under it. Role cards are the part that gives way: one is about 1.7 KB, so
#: 60 of them would pass the clip alone.
HTML_BUDGET = 90_000

#: Kept back for the one notice that says the cards gave way, and the line
#: that says how many roles are left to the digest on disk. Bytes the roles do
#: not spend go to the hidden list.
_BUDGET_RESERVE = 700

_COMPACT_NOTICE = (
    "The roles below are one line each, so this email stays under the size "
    "Gmail clips at. The full cards are in the digest on disk (rolescan show)."
)


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _role_line_html(item: ScoredJob) -> str:
    """A role in one line, for when the cards no longer fit the budget."""
    return (
        f'<div style="{_LI}">{_job_line_html(item.job)} · '
        f"{_BADGE[item.verdict]} {item.score}</div>"
    )


class _RoleBudget:
    """Spends the room the roles have, in rank order: a full card while it
    fits with a line still reserved for every role after it, then one line
    each, then a count of what is left. Once a card does not fit, no later
    role gets one, so the cards are always the top of the ranking, and every
    role has at least a link for as long as the lines themselves fit."""

    def __init__(self, room: int, items: list[ScoredJob]) -> None:
        self.room = room - _BUDGET_RESERVE
        self.pending = sum(_size(_role_line_html(item)) for item in items)
        self.compact = False
        self.left = 0

    def add(self, items: list[ScoredJob]) -> list[str]:
        out: list[str] = []
        for item in items:
            line = _role_line_html(item)
            self.pending -= _size(line)
            if not self.compact:
                card = _role_html(item)
                if _size(card) + self.pending <= self.room:
                    self.room -= _size(card)
                    out.append(card)
                    continue
                self.compact = True
                out.append(f'<div style="{_LEAD}">{_esc(_COMPACT_NOTICE)}</div>')
            if _size(line) <= self.room:
                self.room -= _size(line)
                out.append(line)
            else:
                self.left += 1
        return out

    def rest(self) -> list[str]:
        if not self.left:
            return []
        more = f"{_plural(self.left, 'more role')} in the digest on disk."
        return [f'<div style="{_LEAD}">{_esc(more)}</div>']


def _roles_intro_html(result: ScanResult, view: _HiddenView) -> list[str]:
    """What opens the roles: the stats line, and a lead when there are none."""
    stats = _esc(_stats(result, view))
    if not result.reportable:
        return [
            f'<div style="{_LEAD}">Nothing new worth your time today.</div>',
            f'<div style="{_STATS}">{stats}</div>',
        ]
    live = sum(1 for s in result.reportable if not s.is_blocked)
    blocked = len(result.reportable) - live
    return [
        f'<div style="{_STATS}"><strong>{live} worth a look</strong>'
        + (f", {blocked} blocked" if blocked else "")
        + f". {stats}</div>"
    ]


def _role_cards_html(result: ScanResult, room: int) -> list[str]:
    """The roles, within `room` bytes (see `HTML_BUDGET`)."""
    live = [s for s in result.reportable if not s.is_blocked]
    blocked = [s for s in result.reportable if s.is_blocked]
    out: list[str] = []
    budget = _RoleBudget(room, live + blocked)
    if live:
        out.append(f'<h2 style="{_H2}">Worth a look</h2>')
        out += budget.add(live)
    if blocked:
        out += [
            f'<h2 style="{_H2}">Blocked</h2>',
            f'<div style="{_LEAD}">Structurally closed to you. Listed so you '
            "know the market moved, not as options.</div>",
        ]
        out += budget.add(blocked)
    return out + budget.rest()


def _total(parts: list[str]) -> int:
    return sum(_size(part) for part in parts)


def _fitted(result: ScanResult, view: _HiddenView, rows: int) -> _HiddenView:
    """`view` cut to its `rows` highest-scoring postings: the same choice the
    cap makes, so what the email lists is always among what the cap lists."""
    if rows >= view.listed:
        return view
    cut = _hidden_view(result, rows)
    return replace(cut, file=view.file, fitted=True)


def _fit_hidden(result: ScanResult, view: _HiddenView, room: int) -> _HiddenView:
    """The most rows of `view` whose section costs no more than `room` bytes
    beyond the section with none (its heading, counts and pointers), found by
    halving: a row never makes the section smaller."""
    floor = _total(_rule_hidden_html(_fitted(result, view, 0)))
    low, high = 0, view.listed
    while low < high:
        mid = (low + high + 1) // 2
        cost = _total(_rule_hidden_html(_fitted(result, view, mid))) - floor
        if cost <= room:
            low = mid
        else:
            high = mid - 1
    return _fitted(result, view, low)


def _page(heading: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(heading)}</title></head>"
        f'<body style="{_BODY}">'
        '<table role="presentation" width="100%" cellpadding="0" '
        'cellspacing="0" border="0"><tr><td align="center" '
        'style="padding:16px">'
        '<table role="presentation" width="100%" cellpadding="0" '
        f'cellspacing="0" border="0" style="{_SHELL}">'
        f'<tr><td style="{_CELL}">{body}</td></tr>'
        "</table></td></tr></table></body></html>"
    )


def render_html(
    result: ScanResult,
    *,
    title: str = "Job scan",
    shortlist: list[tuple[str, str, str]] | None = None,
    config_path: Path | None = None,
    hidden_max: int | None = None,
    hidden_file: str = "",
) -> str:
    """The email's HTML part, built from the scan result rather than from the
    markdown. Same inputs as `render_markdown`, and everything that renders
    there renders here.

    Under `HTML_BUDGET` bytes unless the parts that are not roles are already
    over it (2.6.0). The alarms, the unread list and the shortlist are laid out
    first. The role cards spend what is left, then give way to one line each.
    The "Hidden by your rules" section gets what the roles leave: its heading
    and counts always, and as many of its rows as fit, the highest scores
    first."""
    stamp = datetime.now(UTC).strftime("%A %d %B %Y")
    heading = f"{title}, {stamp}"
    view = _hidden_view(result, hidden_max, hidden_file)
    head = [f'<h1 style="{_H1}">{_esc(heading)}</h1>']
    if result.dry_run:
        head.append(f'<div style="{_DRY}">Dry run: nothing was marked as seen.</div>')
    head += _failures_html(result)
    unread = _unread_html(result)
    after = _shortlist_html(shortlist, config_path) if shortlist else []
    # The roles spend first. What the roles do not need is the hidden list's,
    # so a long list can never push a role out of the email; the list keeps its
    # heading, its counts and where the rest is, and as many rows as then fit.
    bare = _total(_rule_hidden_html(_fitted(result, view, 0)))
    fixed = _total([*head, *unread, *after, _page(heading, "")]) + bare
    intro = _roles_intro_html(result, view)  # the widest stats line it can have
    cards = _role_cards_html(result, HTML_BUDGET - fixed - _total(intro))
    room = HTML_BUDGET - fixed - _total(intro) - _total(cards)
    shown = _fit_hidden(result, view, room)
    roles = [*_roles_intro_html(result, shown), *cards]
    return _page(
        heading, "".join([*head, *roles, *unread, *_rule_hidden_html(shown), *after])
    )


def next_digest_path(directory: Path, *, name: str | None = None) -> Path:
    """The file `write_digest` will write, chosen before the digest is
    rendered (2.6.0) so the digest can name the files that sit beside it.

    `name` given: `directory / name`. Otherwise the scan is stamped to the
    minute and suffixed if that minute already has one, so a second scan of the
    day cannot overwrite the first (2026-09-28: four scans, one file). Writes
    nothing."""
    if name is not None:
        return directory / name
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M")
    path = directory / f"{stamp}.md"
    n = 1
    while path.exists():
        n += 1
        path = directory / f"{stamp}-{n}.md"
    return path


def write_digest(
    text: str,
    directory: Path,
    *,
    name: str | None = None,
    path: Path | None = None,
) -> Path:
    """Write the digest atomically; one file per scan.

    `name` given: write only that file (a dry run writes `digest-dry.md` and
    must not become `latest.md`, which `rolescan show` and downstream tooling
    read as the last real run). Otherwise the file is stamped to the minute
    (see `next_digest_path`) and becomes `latest.md` too. `path` is the file a
    caller already chose with `next_digest_path` for the same `name`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = next_digest_path(directory, name=name)
    _write_atomic(path, text)
    if name is None:
        _write_atomic(directory / "latest.md", text)
    return path


def hidden_list_path(digest: Path) -> Path:
    """Where the whole "Hidden by your rules" list goes: beside the digest, with
    its stem and `-hidden` (`2026-10-09T0630.md` -> `2026-10-09T0630-hidden.md`)."""
    return digest.with_name(f"{digest.stem}-hidden{digest.suffix}")


def hidden_list_needed(result: ScanResult, hidden_max: int | None) -> bool:
    """Whether `hidden_max` cuts the list, so the whole of it needs a file."""
    return _hidden_view(result, hidden_max).capped


def write_hidden_list(text: str, digest: Path) -> Path:
    """Write the whole hidden list beside `digest`, atomically."""
    path = hidden_list_path(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, text)
    return path


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class EmailError(RuntimeError):
    """The digest could not be sent for a reason the message explains."""


#: Implicit TLS (SMTPS): the connection is encrypted from its first byte.
#: Every other port starts in plain text and upgrades with STARTTLS.
_IMPLICIT_TLS_PORT = 465

_SMTP_TIMEOUT = 30


def _no_answer(cfg: EmailConfig, cause: Exception) -> str:
    """What to try when the mail port does not answer.

    A VPN tunnel that drops mail ports while HTTPS passes looks, from here, like
    a timeout: nothing says the port is blocked."""
    where = f"mail port {cfg.smtp_port} on {cfg.smtp_host} did not answer"
    if cfg.bind_interface:
        where += f" from interface {cfg.bind_interface}"
    return (
        f"{where}; a VPN or firewall may block mail ports; set "
        "email.bind_interface to the interface that reaches the internet "
        f"directly (for example en0) ({type(cause).__name__}: {cause})"
    )


def _connect(cfg: EmailConfig, source: tuple[str, int] | None) -> smtplib.SMTP:
    """An open, encrypted connection, before login.

    Connect and the TLS handshake are the steps a blocked port stalls, so a
    network error there (and only there) becomes an `EmailError` that says what
    to try. That covers a connect that times out (`TimeoutError`, an `OSError`)
    and a banner that never arrives, which smtplib reports as
    `SMTPServerDisconnected`. Every other `smtplib.SMTPException` (also an
    `OSError`) means the server answered and said no, and so does a certificate
    that does not verify: neither is a blocked port, and each reaches the caller
    as it is.
    """
    # Only name source_address when binding, so the unbound call is exactly the
    # one the module has always made.
    options: dict[str, Any] = {"timeout": _SMTP_TIMEOUT}
    if source is not None:
        options["source_address"] = source
    client: smtplib.SMTP | None = None
    try:
        if cfg.smtp_port == _IMPLICIT_TLS_PORT:
            # smtplib's own default context for SMTP_SSL does not verify the
            # server's certificate; this one does.
            client = smtplib.SMTP_SSL(
                cfg.smtp_host,
                cfg.smtp_port,
                context=ssl.create_default_context(),
                **options,
            )
        else:
            client = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, **options)
            # `starttls()` with no context checks neither the certificate nor
            # the host name, so whoever sits in the path could read the login
            # and the digest. The default context verifies both (2.6.0).
            client.starttls(context=ssl.create_default_context())
    except (smtplib.SMTPException, OSError) as e:
        # close(), not a QUIT: the connection is not in a state to be talked to.
        if client is not None:
            client.close()
        answered = isinstance(e, ssl.SSLCertVerificationError) or (
            isinstance(e, smtplib.SMTPException)
            and not isinstance(e, smtplib.SMTPServerDisconnected)
        )
        if answered:
            raise
        raise EmailError(_no_answer(cfg, e)) from e
    return client


def send_email(
    text: str,
    cfg: EmailConfig,
    *,
    subject: str | None = None,
    html_body: str | None = None,
) -> bool:
    """Send the digest. Named `html_body`, not `html`, because this module
    imports the stdlib `html` for escaping and a parameter of that name
    shadows it: the next person reaching for `html.escape()` in here would get
    an AttributeError on a str.

    A failure raises, and `cli._deliver` turns that into exit status 1 with the
    digest still on disk. With `cfg.bind_interface` set, the interface's address
    is looked up here, at every send, and the socket is bound to it."""
    if not cfg.enabled:
        return False
    source: tuple[str, int] | None = None
    if cfg.bind_interface:
        try:
            source = (interface_ipv4(cfg.bind_interface), 0)
        except ValueError as e:
            problem = f"email.bind_interface: {e}"
            raise EmailError(problem) from e
    msg = EmailMessage()
    msg["Subject"] = subject or f"Job scan {datetime.now(UTC).strftime('%d %b')}"
    msg["From"] = cfg.username
    msg["To"] = cfg.to
    msg.set_content(text)
    if html_body is not None:
        msg.add_alternative(html_body, subtype="html")
    with _connect(cfg, source) as s:
        s.login(cfg.username, cfg.password)
        s.send_message(msg)
    return True
