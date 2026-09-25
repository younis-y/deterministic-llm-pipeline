"""Tests for digest rendering.

`ScanResult` and `SourceReport` live in `rolescan.pipeline`, not
`rolescan.models` — the original task brief pointed at the wrong module and
invented field names (`items`, `failures`, `scanned`) that do not exist on
the real dataclass. The helpers below use the actual fields: `reports`
(a list of `SourceReport`), `reportable`, `unique`, `already_seen`,
`prefiltered`, `llm_calls`, `llm_cached`, `llm_errors`, `dry_run`.
"""

from __future__ import annotations

from pathlib import Path

from rolescan.digest import render_html, render_markdown
from rolescan.models import (
    Confidence,
    CVVariant,
    FitVerdict,
    Job,
    ScoredJob,
    Verdict,
)
from rolescan.pipeline import ScanResult, SourceReport


def _empty_result() -> ScanResult:
    """A completed scan that matched nothing and had no failures."""
    return ScanResult(
        reports=[SourceReport(kind="lever", slug="acme", label="Acme", count=0)]
    )


def _all_sources_failed_result() -> ScanResult:
    """Every configured source raised. Distinct from an empty market."""
    return ScanResult(
        reports=[
            SourceReport(
                kind="linkedin", slug="x", label="LinkedIn", error="challenge page"
            ),
            SourceReport(kind="adzuna", slug="y", label="Adzuna", error="401"),
        ]
    )


def _job() -> Job:
    return Job(
        source="greenhouse",
        company="Masdar",
        title="Data Scientist",
        location="Abu Dhabi",
        url="https://x/7",
        description="Analytics. UAE National (National Talent programme).",
    )


def _fit(**kw: object) -> FitVerdict:
    base: dict[str, object] = {
        "fit_score": 78,
        "verdict": Verdict.APPLY,
        "confidence": Confidence.HIGH,
        "reason": "Strong analytics overlap.",
        "cv_variant": CVVariant.DATA_SCIENCE,
        "tailoring": ["Lead with the forecasting project."],
        "blockers": [],
        "keywords_missing": [],
    }
    base.update(kw)
    return FitVerdict.model_validate(base)


# --- a blocked role must say why, and must not invite an application -------


def test_a_blocked_role_names_the_configured_term_the_llm_missed() -> None:
    """The case the whole blocker change turns on: the configured term
    blocked this posting and the model itself noticed nothing, so
    `fit.blockers` is empty. Render the merged list or the digest shows a
    BLOCKED badge with no reason under it, which reads as a bug in the tool
    rather than a fact about the role."""
    item = ScoredJob(
        job=_job(),
        keyword_score=30,
        blocker_hits=["uae national"],
        fit=_fit(blockers=[]),
    )
    text = render_markdown(ScanResult(reportable=[item]))
    assert "`BLOCKED`" in text
    assert "**Blocked by:** uae national" in text
    assert "**Send:**" not in text, "no CV advice for a role you cannot be given"
    assert "**Tailor it:**" not in text


def test_a_blocked_role_shows_a_reason_on_the_keyword_only_path() -> None:
    """Same rule with no LLM verdict at all. A hard blocker need not carry a
    weight, so it need not appear in keyword_penalties either - and then
    nothing else on the entry names it."""
    item = ScoredJob(job=_job(), keyword_score=30, blocker_hits=["uae national"])
    text = render_markdown(ScanResult(reportable=[item]))
    assert "`BLOCKED`" in text
    assert "**Blocked by:** uae national" in text
    assert "**Send:**" not in text


def test_the_stats_line_counts_blocked_and_hidden_postings() -> None:
    """With show_blocked false the posting is deleted and recorded as seen.
    This clause is the reader's only evidence it existed."""
    text = render_markdown(ScanResult(unique=12, hidden_blocked=2))
    assert "2 blocked and hidden" in text
    assert "show_blocked" in text


def test_the_stats_line_stays_quiet_when_nothing_was_hidden() -> None:
    """A permanent "0 blocked and hidden" would train the reader to skip the
    line on the day it matters."""
    assert "blocked and hidden" not in render_markdown(ScanResult(unique=12))


def test_digest_has_a_shortlist_section_when_given_one() -> None:
    result = _empty_result()
    text = render_markdown(
        result, shortlist=[("https://x/1", "Glencore", "Analytics Graduate Programme")]
    )
    assert "Shortlist" in text
    assert "Glencore" in text
    assert "rolescan mark https://x/1 applied" in text


def test_digest_omits_the_shortlist_section_when_empty() -> None:
    text = render_markdown(_empty_result(), shortlist=[])
    assert "Shortlist" not in text


def test_digest_omits_the_shortlist_section_when_not_given() -> None:
    text = render_markdown(_empty_result())
    assert "Shortlist" not in text


def test_digest_says_so_when_every_source_failed() -> None:
    result = _all_sources_failed_result()
    text = render_markdown(result)
    assert "every source failed" in text.casefold()


def test_digest_does_not_claim_failure_for_a_genuinely_empty_market() -> None:
    text = render_markdown(_empty_result())
    assert "every source failed" not in text.casefold()


def test_digest_does_not_call_a_skipped_source_a_failure() -> None:
    """Skipped (not configured, e.g. Adzuna with no API key) is not the same
    as errored. Calling it a failure sends the reader to debug a working
    scraper at 06:30 instead of ignoring an expected config gap."""
    result = ScanResult(
        reports=[
            SourceReport(
                kind="adzuna", slug="x", label="Adzuna", skipped=True, error="no API key"
            ),
            SourceReport(
                kind="linkedin",
                slug="y",
                label="LinkedIn",
                skipped=True,
                error="unsupported region",
            ),
        ]
    )
    text = render_markdown(result)
    assert "failed" not in text.casefold()
    assert "no configured source was able to run" in text.casefold()


def test_digest_reports_a_mix_of_errors_and_skips_truthfully() -> None:
    result = ScanResult(
        reports=[
            SourceReport(kind="adzuna", slug="x", label="Adzuna", skipped=True, error="no API key"),
            SourceReport(kind="linkedin", slug="y", label="LinkedIn", error="challenge page"),
        ]
    )
    text = render_markdown(result)
    lowered = text.casefold()
    assert "1 source(s) errored" in lowered
    assert "1 declined to run" in lowered
    # Must not claim every source failed when one only declined to run.
    assert "every source failed" not in lowered


def test_shortlist_row_with_blank_company_and_title_falls_back_to_url() -> None:
    """The Task 6 defect: `mark shortlist` on an unscanned url leaves company
    and title blank. The digest must never render an empty bullet for it —
    it must fall back to something a human can act on, here the url."""
    text = render_markdown(
        _empty_result(), shortlist=[("https://x/9", "", "")]
    )
    assert "- https://x/9" in text
    assert "**" + " — " + "**" not in text
    assert "-  \n" not in text


def test_shortlist_row_with_only_company_still_renders() -> None:
    text = render_markdown(_empty_result(), shortlist=[("https://x/2", "Drax", "")])
    assert "Drax" in text
    assert "https://x/2" in text


# --- the HTML part is HTML, not markdown in a box --------------------------


def _one(job: Job | None = None, fit: FitVerdict | None = None) -> ScanResult:
    """A scan carrying one scored role, so a block can be inspected."""
    return ScanResult(
        reportable=[
            ScoredJob(job=job or _job(), keyword_score=40, fit=fit or _fit())
        ]
    )


def test_html_render_wraps_the_body() -> None:
    html = render_html(_empty_result())
    assert html.startswith("<!doctype html>")
    assert "Job scan" in html


def test_the_apply_link_is_a_real_link_not_markdown_source() -> None:
    """The complaint the rewrite exists for. The HTML part used to be the
    markdown escaped into a <pre>, so `[Apply](https://x/7)` arrived as those
    literal characters and the one thing the reader needs was not clickable."""
    html = render_html(_one())
    assert 'href="https://x/7"' in html
    assert "[Apply](" not in html
    assert "**" not in html, "markdown emphasis must not reach the reader"
    assert "<pre" not in html


def test_the_job_title_is_itself_the_link() -> None:
    """The primary action, and the most prominent thing in the block."""
    html = render_html(_one())
    title = html.index("Data Scientist")
    assert 'href="https://x/7"' in html[:title], "the title sits inside the anchor"


def test_html_render_escapes_hostile_content() -> None:
    """Job titles and company names are scraped from third-party pages and
    must be treated as hostile input, not trusted text."""
    job = _job().model_copy(update={"title": "<script>alert(1)</script>"})
    html = render_html(_one(job=job))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_every_scraped_field_is_escaped_not_only_the_title() -> None:
    job = _job().model_copy(
        update={"company": "<b>Ac</b>me", "location": "<i>Dubai</i>"}
    )
    html = render_html(_one(job=job, fit=_fit(reason="<img src=x onerror=1>")))
    for hostile in ("<b>", "<i>", "<img"):
        assert hostile not in html
    assert "&lt;img src=x onerror=1&gt;" in html


def test_a_javascript_url_never_becomes_a_live_link() -> None:
    """Urls are scraped too. The scheme check is an allowlist, not a
    `javascript:` denylist, so an encoding trick has nothing to beat."""
    job = _job().model_copy(update={"url": "javascript:alert(1)"})
    html = render_html(_one(job=job))
    assert "href=" not in html, "nothing on this card may be a link"
    assert "javascript:alert(1)</div>" in html, "shown as inert text instead"


def test_a_url_cannot_break_out_of_the_href_attribute() -> None:
    """`quote=True` is the whole reason the escape helper exists: without it
    a quote in the url closes the attribute and the rest becomes markup."""
    job = _job().model_copy(
        update={"url": 'https://x/7"onmouseover="alert(1)'}
    )
    html = render_html(_one(job=job))
    assert 'onmouseover="alert(1)' not in html
    assert "&quot;onmouseover=&quot;alert(1)" in html


def test_a_hostile_shortlist_url_is_not_a_live_link_either() -> None:
    html = render_html(
        _empty_result(), shortlist=[("javascript:alert(1)", "Acme", "Analyst")]
    )
    assert "href=" not in html


def test_the_html_survives_a_mail_client() -> None:
    """Inline styles only. Stylesheets, web fonts and scripts are stripped by
    most clients, and flexbox and grid are unreliable in several."""
    html = render_html(_one(), shortlist=[("https://x/1", "Acme", "Analyst")])
    for banned in ("<script", "<link", "<style", "@import", "flex", "grid", "onclick"):
        assert banned not in html.casefold()


def test_the_four_verdicts_do_not_look_alike() -> None:
    """A badge the reader has to actually read is not a badge."""
    seen = set()
    for verdict in (Verdict.APPLY, Verdict.CONSIDER, Verdict.SKIP):
        html = render_html(_one(fit=_fit(verdict=verdict)))
        assert f">{verdict.value.upper()}<" in html
        seen.add(html[html.index("border-left:") : html.index("border-left:") + 30])
    blocked = render_html(
        ScanResult(
            reportable=[
                ScoredJob(job=_job(), blocker_hits=["uae national"], fit=_fit())
            ]
        )
    )
    seen.add(blocked[blocked.index("border-left:") : blocked.index("border-left:") + 30])
    assert len(seen) == 4, "each verdict needs its own colour"


def test_a_blocked_role_is_not_invited_to_apply_in_html() -> None:
    result = ScanResult(
        reportable=[
            ScoredJob(job=_job(), blocker_hits=["uae national"], fit=_fit(blockers=[]))
        ]
    )
    html = render_html(result)
    assert ">BLOCKED<" in html
    assert "uae national" in html
    assert ">Apply<" not in html, "no invitation to a role you cannot be given"
    assert ">View posting<" in html, "the posting is still reachable"
    assert "Send:" not in html
    assert "Tailor it" not in html


def test_the_keyword_only_path_renders_in_html() -> None:
    result = ScanResult(
        reportable=[
            ScoredJob(
                job=_job(),
                keyword_score=44,
                keyword_hits=["energy", "python"],
                keyword_penalties=["principal"],
            )
        ]
    )
    html = render_html(result)
    assert "keyword only" in html
    assert "energy, python" in html
    assert "principal" in html
    assert ">44<" in html


def test_the_tailoring_advice_renders_but_subordinate() -> None:
    html = render_html(_one())
    assert "CV_DataScience-Gulf" in html
    assert "Lead with the forecasting project." in html
    assert html.index("Strong analytics overlap.") < html.index("Tailor it"), (
        "the reason comes before the tailoring the reader may never open"
    )


def test_html_keeps_the_stats_line_and_the_hidden_blocked_count() -> None:
    html = render_html(ScanResult(unique=12, hidden_blocked=2))
    assert "Scanned 12 unique postings" in html
    assert "2 blocked and hidden" in html


def test_html_keeps_the_failures_and_skipped_blocks() -> None:
    result = ScanResult(
        reports=[
            SourceReport(kind="lever", slug="acme", label="Acme", error="timeout"),
            SourceReport(
                kind="adzuna", slug="x", label="Adzuna", skipped=True, error="no API key"
            ),
        ]
    )
    html = render_html(result)
    assert "Sources that failed this run" in html
    assert "lever/acme" in html
    assert "timeout" in html
    assert "Sources skipped (not searched)" in html
    assert "rolescan discover" in html


def test_html_keeps_the_unusable_backend_note() -> None:
    html = render_html(ScanResult(llm_unusable="ollama is not running"))
    assert "LLM scoring did not run at all" in html
    assert "ollama is not running" in html
    assert "min_report_score" in html
    assert "It is not a quiet market." in html


def test_html_keeps_the_backend_specific_error_hint() -> None:
    html = render_html(
        ScanResult(
            llm_backend="ollama",
            llm_errors=2,
            llm_error_detail="RuntimeError: ollama returned HTTP 500",
        )
    )
    assert "ANTHROPIC_API_KEY" not in html
    assert "needs no API key" in html
    assert "llm.timeout" in html
    assert "ollama returned HTTP 500" in html
    assert "ANTHROPIC_API_KEY" in render_html(
        ScanResult(llm_backend="anthropic", llm_errors=1)
    )


def test_html_keeps_the_shortlist_and_its_commands() -> None:
    html = render_html(
        _empty_result(),
        shortlist=[("https://x/1", "Glencore", "Analytics Graduate Programme")],
        config_path=Path("/etc/rolescan/config.yaml"),
    )
    assert "Shortlist" in html
    assert "Glencore" in html
    assert "rolescan mark https://x/1 applied --config /etc/rolescan/config.yaml" in html
    assert "rolescan mark https://x/1 dismissed" in html


def test_html_shortlist_row_with_no_company_or_title_falls_back_to_the_url() -> None:
    html = render_html(_empty_result(), shortlist=[("https://x/9", "", "")])
    assert "https://x/9" in html


def test_html_says_nothing_new_when_there_is_nothing_new() -> None:
    html = render_html(_empty_result())
    assert "Nothing new worth your time today." in html
    assert "Worth a look" not in html


def test_html_marks_a_dry_run() -> None:
    html = render_html(ScanResult(dry_run=True))
    assert "Dry run" in html


def test_html_is_one_column_and_fits_a_phone() -> None:
    html = render_html(_one())
    assert "width=device-width" in html
    assert "max-width:620px" in html
    assert "word-break:break-word" in html


# --- the failure notes have to fit the backend that actually failed --------


def test_an_ollama_error_does_not_blame_the_anthropic_key() -> None:
    """`ollama returned HTTP 500` has nothing to do with a credential that
    backend has never had. The wave generalised this string in judges.py and
    cli.py and missed the one place the end user actually reads."""
    text = render_markdown(
        ScanResult(
            llm_backend="ollama",
            llm_errors=2,
            llm_error_detail="RuntimeError: ollama returned HTTP 500",
        )
    )
    assert "ANTHROPIC_API_KEY" not in text
    assert "needs no API key" in text
    assert "llm.timeout" in text


def test_an_anthropic_error_still_names_the_key() -> None:
    text = render_markdown(ScanResult(llm_backend="anthropic", llm_errors=1))
    assert "ANTHROPIC_API_KEY" in text
    assert "401" in text


def test_an_unregistered_backend_loses_the_hint_but_not_the_layout() -> None:
    """`_llm_error_hint` returns "" for a backend nothing registered, and the
    line was appended regardless, leaving a stray blank line where the advice
    should be. run_scan always sets llm_backend, so no scan reaches this; a
    ScanResult built anywhere else does, and the failure section is the one
    place the reader is told what to do next."""
    text = render_markdown(
        ScanResult(
            llm_backend="",
            llm_errors=1,
            llm_error_detail="RuntimeError: something went wrong",
        )
    )
    assert "LLM scoring failed for 1 posting(s)" in text
    assert text.endswith("`RuntimeError: something went wrong`\n"), (
        "the section ends at the detail, with no blank line standing in for "
        "advice that was never produced"
    )


def test_a_failed_preflight_does_not_claim_nothing_was_scored() -> None:
    """The 3s liveness probe covers connect, read, write and pool, and
    `FitScorer` is built regardless of it - so a slow `/api/tags` produced a
    digest that printed "LLM scoring did not run at all" above postings
    carrying fit scores and confidence levels. A message that contradicts the
    page it is printed on teaches the reader to ignore it."""
    result = ScanResult(
        llm_backend="ollama",
        llm_unusable="could not reach ollama at http://localhost:11434",
        llm_calls=4,
        llm_cached=1,
    )
    text = render_markdown(result)
    assert "did not run at all" not in text
    assert "scoring ran anyway" in text


def test_a_backend_that_really_did_not_run_still_says_so() -> None:
    result = ScanResult(
        llm_backend="ollama",
        llm_unusable="could not reach ollama at http://localhost:11434",
    )
    text = render_markdown(result)
    assert "LLM scoring did not run at all" in text
