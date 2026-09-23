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
        if fit.blockers:
            lines.append("**Blocked by:** " + "; ".join(fit.blockers))
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


def _shortlist_section(shortlist: list[tuple[str, str, str]]) -> list[str]:
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
            f"  `rolescan mark {url} applied` · `rolescan mark {url} dismissed`"
        )
    return lines


def render_markdown(
    result: ScanResult,
    *,
    title: str = "Job scan",
    shortlist: list[tuple[str, str, str]] | None = None,
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
            out += _shortlist_section(shortlist)
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
        out += _shortlist_section(shortlist)
    return "\n".join(out)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _every_source_failed(result: ScanResult) -> bool:
    """True only when sources actually ran and none of them succeeded.

    Distinct from an empty market: zero reportable roles because nothing
    matched looks identical, in the digest, to zero because the scan never
    fetched anything. This is what tells the two apart.
    """
    return bool(result.reports) and not any(r.ok for r in result.reports)


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
    text = ". ".join(bits) + "."
    if _every_source_failed(result):
        text += (
            " Every source failed this run. This is a scan failure, not an "
            "empty market."
        )
    return text


def _failures(result: ScanResult) -> list[str]:
    failed = result.failed_sources
    skipped = result.skipped_sources
    if not failed and not skipped and not result.llm_errors:
        return []
    lines = ["", "---", ""]
    if result.llm_errors:
        # Every score fell back to keywords. Without this the digest is
        # indistinguishable from a deliberate --no-llm run.
        lines += [
            f"**LLM scoring failed for {result.llm_errors} posting(s)** "
            "— those roles are ranked on keyword score alone.",
            "",
            f"`{result.llm_error_detail}`" if result.llm_error_detail else "",
            "",
            "A 401 here means ANTHROPIC_API_KEY is missing, revoked, or from "
            "another organisation.",
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
    html: str | None = None,
) -> bool:
    if not cfg.enabled:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject or f"Job scan {datetime.now(UTC).strftime('%d %b')}"
    msg["From"] = cfg.username
    msg["To"] = cfg.to
    msg.set_content(text)
    if html is not None:
        msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as s:
        s.starttls()
        s.login(cfg.username, cfg.password)
        s.send_message(msg)
    return True
