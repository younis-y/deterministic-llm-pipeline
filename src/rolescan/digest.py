"""Digest rendering and delivery."""

from __future__ import annotations

import html
import logging
import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path

from rolescan.config import EmailConfig
from rolescan.models import ScoredJob, Verdict
from rolescan.pipeline import ScanResult
from rolescan.scoring.judges import available_judges

__all__ = ["render_html", "render_markdown", "send_email", "write_digest"]

log = logging.getLogger(__name__)

_BADGE = {
    Verdict.APPLY: "APPLY",
    Verdict.CONSIDER: "CONSIDER",
    Verdict.SKIP: "SKIP",
    Verdict.BLOCKED: "BLOCKED",
}


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
        # No CV advice for a role you cannot be considered for. Suggesting one
        # reads as an invitation to waste an afternoon on it.
        if not item.is_blocked:
            lines.append(f"**Send:** {fit.cv_variant.value}")
            lines.append("")
            if fit.tailoring:
                lines.append("**Tailor it:**")
                lines.extend(f"- {t}" for t in fit.tailoring)
                lines.append("")
            if fit.keywords_missing:
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
    if result.llm_calls or result.llm_cached:
        bits.append(f"{result.llm_calls} scored, {result.llm_cached} from cache")
    if result.hidden_blocked:
        # Only when it happened: a permanent "0 blocked and hidden" on every
        # digest would train the reader to skip the line that matters. These
        # postings scored well enough to be here and were removed anyway, and
        # they are already recorded as seen, so this is the reader's only
        # chance to notice a blocker term that is matching the wrong thing.
        bits.append(
            f"{result.hidden_blocked} blocked and hidden "
            "(set output.show_blocked true to see them)"
        )
    return ". ".join(bits) + "." + _run_outcome_note(result)


def _llm_ran(result: ScanResult) -> bool:
    """Whether any posting in this digest actually carries an LLM score."""
    return bool(result.llm_calls or result.llm_cached)


def _llm_error_hint(result: ScanResult) -> str:
    """One line naming the likely cause, for the backend that was configured.

    "A 401 here means ANTHROPIC_API_KEY is missing" is the right sentence for
    the hosted backend and exactly the wrong one for `ollama returned HTTP
    500`: it sends someone to look at a credential that backend has never
    had. This is the place the end user actually reads, so the distinction
    has to be real here, not only in the CLI.
    """
    judge = available_judges().get(result.llm_backend)
    if judge is None:
        return ""
    if judge.needs_api_key:
        env = judge.api_key_env or "the backend's API key"
        return (
            f"A 401 here means {env} is missing, revoked, or from another "
            "organisation."
        )
    return (
        f"The `{result.llm_backend}` backend needs no API key, so a credential "
        "is not the cause. Check the server is still up and still holding "
        "`llm.model`, and that `llm.timeout` is long enough for it."
    )


def _failures(result: ScanResult) -> list[str]:
    failed = result.failed_sources
    skipped = result.skipped_sources
    if not failed and not skipped and not result.llm_errors and not result.llm_unusable:
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
            "Every posting in this digest was ranked on keyword score alone. "
            "Keyword scores are not on the same scale as the fit scores "
            "`min_report_score` was set for, so expect this digest to be much "
            "shorter than it should be — or empty — until the backend works. "
            "It is not a quiet market.",
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
            "The scores below are real. It is the liveness probe that failed, "
            "not the backend — it is deliberately short so an unattended run "
            "cannot hang on it, and a loaded server can exceed it.",
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
            _llm_error_hint(result),
            "",
        ]
    if failed:
        lines += ["**Sources that failed this run**", ""]
        lines += [f"- `{r.kind}/{r.slug}` {r.error}" for r in failed]
        lines += ["", "Run `rolescan discover` to check the slugs.", ""]
    if skipped:
        # Called out separately because a skipped source contributed nothing
        # and is easy to mistake for one that found nothing.
        lines += ["**Sources skipped (not searched)**", ""]
        lines += [f"- `{r.kind}/{r.slug}` {r.error}" for r in skipped]
        lines += [""]
    return lines


def render_html(text: str) -> str:
    """Wrap the markdown body in a minimal document that reads on a phone.

    Deliberately not a markdown-to-HTML converter: the digest is read at
    06:30 on a small screen, and a <pre> block with a sane font beats a
    dependency. `text` may contain job titles and company names scraped from
    third-party pages, so it is escaped before going anywhere near HTML.
    """
    escaped = html.escape(text)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "</head><body style='margin:0;padding:16px;background:#fff;color:#111;"
        "font:15px/1.5 -apple-system,BlinkMacSystemFont,sans-serif'>"
        f"<pre style='white-space:pre-wrap;font:inherit;margin:0'>{escaped}</pre>"
        "</body></html>"
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
