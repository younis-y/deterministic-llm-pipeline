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
