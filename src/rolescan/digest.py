"""Digest rendering and delivery."""

from __future__ import annotations

import html
import logging
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path

from rolescan.config import EmailConfig
from rolescan.models import FitVerdict, Job, ScoredJob, Verdict
from rolescan.pipeline import ScanResult, SourceReport
from rolescan.scoring.judges import available_judges
from rolescan.scoring.rules import RULE_ORDER

__all__ = ["render_html", "render_markdown", "send_email", "write_digest"]

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


def _hidden_group(item: ScoredJob) -> tuple[str, str] | None:
    """The group a rule-hidden posting is listed under, and its reason.

    The rule `decide` fired wins over a `hard_blockers` term, so a posting
    both caught is listed once, under the rule: its reason quotes the advert,
    which is what a reader checks a skip against. A judge-mode model block
    has no rule (`decide` never ran) and goes under `hard_bar` with the
    model's reason, ahead of any term it also matched. A term-only posting's
    reason is the term(s) that matched, as configured.
    """
    fit = item.fit
    if fit is not None and fit.rule is not None:
        return fit.rule, _hidden_reason(fit.reason)
    if fit is not None and fit.verdict is Verdict.BLOCKED:
        return "hard_bar", _hidden_reason(fit.reason)
    if item.blocker_hits:
        terms = ", ".join(f'"{t}"' for t in dict.fromkeys(item.blocker_hits))
        return _TERMS, f"blocked by {terms}"
    return None


def _rule_hidden_groups(result: ScanResult) -> list[tuple[str, list[tuple[Job, str]]]]:
    """(heading, [(posting, reason)]) per rule, in `decide`'s order, then the
    `hard_blockers` terms group.

    A rule name this module has no label for (one added to `decide` without
    updating `_RULE_LABELS`) still renders, under its own name and after the
    known rules, rather than vanishing from the one place it is reported.
    """
    groups: dict[str, list[tuple[Job, str]]] = {}
    for item in result.rule_hidden:
        if (found := _hidden_group(item)) is None:
            continue
        group, reason = found
        groups.setdefault(group, []).append((item.job, reason))
    rank = {rule: i for i, rule in enumerate(RULE_ORDER)}
    rank[_TERMS] = len(RULE_ORDER) + 1
    return [
        (
            _RULE_LABELS.get(rule, rule),
            sorted(
                groups[rule],
                key=lambda row: (row[0].company.casefold(), row[0].title.casefold()),
            ),
        )
        for rule in sorted(groups, key=lambda r: (rank.get(r, len(RULE_ORDER)), r))
    ]


def _rule_hidden_section(result: ScanResult) -> list[str]:
    groups = _rule_hidden_groups(result)
    if not groups:
        return []
    lines = ["## Hidden by your rules", "", _RULE_HIDDEN_LEAD, ""]
    for heading, rows in groups:
        lines += [f"**{heading}**", ""]
        for job, reason in rows:
            bits = [f"**{job.company}**", f"[{job.title}]({job.url})"]
            if job.location:
                bits.append(job.location)
            lines.append(f"- {' · '.join(bits)}: {reason}")
        lines.append("")
    return lines


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
) -> str:
    stamp = datetime.now(UTC).strftime("%A %d %B %Y")
    out: list[str] = [f"# {title}, {stamp}", ""]

    if result.dry_run:
        out += ["*Dry run: nothing was marked as seen.*", ""]

    if not result.reportable:
        out += [
            "Nothing new worth your time today.",
            "",
            _stats(result),
            "",
        ]
        out += _rule_hidden_section(result)
        out += _failures(result)
        if shortlist:
            out += _shortlist_section(shortlist, config_path)
        return "\n".join(out)

    live = [s for s in result.reportable if not s.is_blocked]
    blocked = [s for s in result.reportable if s.is_blocked]

    out.append(
        f"**{len(live)} worth a look**"
        + (f", {len(blocked)} blocked" if blocked else "")
        + f". {_stats(result)}"
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

    out += _rule_hidden_section(result)
    out += _failures(result)
    if shortlist:
        out += _shortlist_section(shortlist, config_path)
    return "\n".join(out)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


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


def _stats(result: ScanResult) -> str:
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
    if result.llm_calls or result.llm_cached:
        bits.append(f"{result.llm_calls} scored, {result.llm_cached} from cache")
    if result.hidden_blocked:
        # Only when it happened: a permanent "0 blocked and hidden" on every
        # digest would train the reader to skip the line that matters. These
        # postings scored well enough to be here and were removed anyway, and
        # they are already recorded as seen, so this is the reader's only
        # chance to notice a blocker term that is matching the wrong thing.
        bits.append(
            f"{result.hidden_blocked} blocked and hidden (output.show_blocked is false)"
        )
    if listed := sum(len(rows) for _, rows in _rule_hidden_groups(result)):
        # Only when it happened, for the same reason. Counted separately from
        # the clause above: that one is postings `show_blocked` removed, this
        # one is postings any rule or `hard_blockers` term removed by any
        # route, so a block hidden by `show_blocked` is in both - hence
        # "listed below", so the two numbers do not read as two postings. It
        # counts the rows the section renders, so the number and the list
        # can never disagree.
        bits.append(f"{listed} hidden by your rules (listed below)")
    return ". ".join(bits) + "." + _run_outcome_note(result)


def _llm_ran(result: ScanResult) -> bool:
    """Whether any posting in this digest actually carries an LLM score."""
    return bool(result.llm_calls or result.llm_cached)


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


def _failures(result: ScanResult) -> list[str]:
    failed = result.failed_sources
    skipped = result.skipped_sources
    if (
        not failed
        and not skipped
        and not result.llm_errors
        and not result.llm_unusable
        and not result.enricher_unusable
    ):
        return []
    lines = ["", "---", ""]
    if result.llm_unusable and not _llm_ran(result):
        # The configured judge never ran at all, so there are no scoring errors
        # to report and the digest would otherwise look like a normal quiet
        # day. It is not one: every posting fell back to a keyword score, and
        # keyword scores are not calibrated against min_report_score, so this
        # digest is close to empty by construction rather than by market.
        lines += [
            f"**LLM scoring did not run at all.** {result.llm_unusable}.",
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
    return lines


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
    # option - the same reason CV advice was withheld for one.
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
                f"LLM scoring did not run at all. {result.llm_unusable}.",
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
    return out


def _failures_html(result: ScanResult) -> list[str]:
    failed = result.failed_sources
    skipped = result.skipped_sources
    if (
        not failed
        and not skipped
        and not result.quiet_sources
        and not result.llm_errors
        and not result.llm_unusable
        and not result.enricher_unusable
    ):
        return []
    out = [f'<hr style="{_RULE}">', *_llm_notes_html(result)]
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
    return out


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


def _rule_hidden_line_html(job: Job, reason: str) -> str:
    title = _esc(job.title)
    if href := _href(job.url):
        title = f'<a href="{href}" style="{_LINK}">{title}</a>'
    bits = [f"<strong>{_esc(job.company)}</strong>", title]
    if job.location:
        bits.append(_esc(job.location))
    return f"{' · '.join(bits)}: {_esc(reason)}"


def _rule_hidden_html(result: ScanResult) -> list[str]:
    """The HTML counterpart of `_rule_hidden_section`."""
    groups = _rule_hidden_groups(result)
    if not groups:
        return []
    out = [
        f'<h2 style="{_H2}">Hidden by your rules</h2>',
        f'<div style="{_LEAD}">{_esc(_RULE_HIDDEN_LEAD)}</div>',
    ]
    for heading, rows in groups:
        items = "".join(
            f'<li style="{_LI}">{_rule_hidden_line_html(job, reason)}</li>'
            for job, reason in rows
        )
        out += [
            f'<div style="{_SUB_LABEL}">{_esc(heading)}</div>',
            f'<ul style="{_UL}">{items}</ul>',
        ]
    return out


def _roles_html(result: ScanResult) -> list[str]:
    if not result.reportable:
        return [
            f'<div style="{_LEAD}">Nothing new worth your time today.</div>',
            f'<div style="{_STATS}">{_esc(_stats(result))}</div>',
        ]
    live = [s for s in result.reportable if not s.is_blocked]
    blocked = [s for s in result.reportable if s.is_blocked]
    out = [
        f'<div style="{_STATS}"><strong>{len(live)} worth a look</strong>'
        + (f", {len(blocked)} blocked" if blocked else "")
        + f". {_esc(_stats(result))}</div>"
    ]
    if live:
        out.append(f'<h2 style="{_H2}">Worth a look</h2>')
        out += [_role_html(item) for item in live]
    if blocked:
        out += [
            f'<h2 style="{_H2}">Blocked</h2>',
            f'<div style="{_LEAD}">Structurally closed to you. Listed so you '
            "know the market moved, not as options.</div>",
        ]
        out += [_role_html(item) for item in blocked]
    return out


def render_html(
    result: ScanResult,
    *,
    title: str = "Job scan",
    shortlist: list[tuple[str, str, str]] | None = None,
    config_path: Path | None = None,
) -> str:
    """The email's HTML part, built from the scan result rather than from the
    markdown. Same inputs as `render_markdown`, and everything that renders
    there renders here."""
    stamp = datetime.now(UTC).strftime("%A %d %B %Y")
    heading = f"{title}, {stamp}"
    body = [f'<h1 style="{_H1}">{_esc(heading)}</h1>']
    if result.dry_run:
        body.append(f'<div style="{_DRY}">Dry run: nothing was marked as seen.</div>')
    body += _roles_html(result)
    body += _rule_hidden_html(result)
    body += _failures_html(result)
    if shortlist:
        body += _shortlist_html(shortlist, config_path)
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
        f'<tr><td style="{_CELL}">{"".join(body)}</td></tr>'
        "</table></td></tr></table></body></html>"
    )


def write_digest(text: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = directory / f"{stamp}.md"
    path.write_text(text, encoding="utf-8")
    (directory / "latest.md").write_text(text, encoding="utf-8")
    return path


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
    an AttributeError on a str."""
    if not cfg.enabled:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject or f"Job scan {datetime.now(UTC).strftime('%d %b')}"
    msg["From"] = cfg.username
    msg["To"] = cfg.to
    msg.set_content(text)
    if html_body is not None:
        msg.add_alternative(html_body, subtype="html")
    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as s:
        s.starttls()
        s.login(cfg.username, cfg.password)
        s.send_message(msg)
    return True
