"""Tests for digest rendering.

`ScanResult` and `SourceReport` live in `rolescan.pipeline`, not
`rolescan.models` — the original task brief pointed at the wrong module and
invented field names (`items`, `failures`, `scanned`) that do not exist on
the real dataclass. The helpers below use the actual fields: `reports`
(a list of `SourceReport`), `reportable`, `unique`, `already_seen`,
`prefiltered`, `llm_calls`, `llm_cached`, `llm_errors`, `dry_run`.
"""

from __future__ import annotations

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


def test_html_render_wraps_the_body() -> None:
    html = render_html("# Job scan\n\nsomething")
    assert html.startswith("<!doctype html>")
    assert "Job scan" in html


def test_html_render_escapes_hostile_content() -> None:
    """Job titles and company names are scraped from third-party pages and
    must be treated as hostile input, not trusted text."""
    html = render_html("<script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


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
