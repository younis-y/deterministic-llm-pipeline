"""The digest names sources that were cut short, shrank, or left a note
(2.5.8), in both renderers, even when nothing else went wrong: the rule the
quiet alarm set in 2.5.7, after the Markdown digest dropped it whenever it
was the only problem."""

from __future__ import annotations

import pytest

from rolescan.digest import render_html, render_markdown
from rolescan.pipeline import ScanResult, SourceReport

CUT = ScanResult(
    truncated_sources=[
        ("Acme", 500, 2000, "stopped at 500 of 2000 postings (max_rows 500)"),
        ("Zenko", 60, None, "page 4 not read (HTTP 500); kept pages 1-3"),
    ]
)
SHRUNK = ScanResult(shrunk_sources=[("Acme", 10, 400.0), ("Zenko", 3, 12.5)])
NOTED = ScanResult(
    reports=[
        SourceReport(
            "workday",
            "acme",
            "Acme",
            9,
            note="skipped 1 posting(s) that are not valid jobs",
        )
    ]
)


def test_markdown_lists_sources_that_were_cut_short_when_nothing_else_failed() -> None:
    text = render_markdown(CUT)
    assert "**Sources that were cut short**" in text
    assert (
        "- **Acme** (500 of 2000 read): stopped at 500 of 2000 postings (max_rows 500)"
        in text
    )
    assert "- **Zenko** (60 read): page 4 not read (HTTP 500); kept pages 1-3" in text
    assert "`max_rows`" in text


def test_markdown_lists_sources_that_shrank_when_nothing_else_failed() -> None:
    text = render_markdown(SHRUNK)
    assert "**Sources that shrank**" in text
    assert "- **Acme** returned 10, under 30% of its recent median of 400" in text
    assert "- **Zenko** returned 3, under 30% of its recent median of 12.5" in text


def test_markdown_lists_notes_when_nothing_else_failed() -> None:
    text = render_markdown(NOTED)
    assert "**Notes**" in text
    assert "- **Acme**: skipped 1 posting(s) that are not valid jobs" in text


@pytest.mark.parametrize(
    ("result", "heading", "line"),
    [
        (CUT, "Sources that were cut short", "(500 of 2000 read): stopped at 500"),
        (SHRUNK, "Sources that shrank", "under 30% of its recent median of 400"),
        (NOTED, "Notes", ": skipped 1 posting(s) that are not valid jobs"),
    ],
)
def test_html_shows_each_section_on_its_own(
    result: ScanResult, heading: str, line: str
) -> None:
    html = render_html(result)
    assert heading in html and line in html


def test_html_escapes_what_a_source_said() -> None:
    result = ScanResult(
        truncated_sources=[("A<b>", 1, 2, "<script>x</script>")],
        shrunk_sources=[("S&P", 1, 40.0)],
        reports=[SourceReport("x", "y", "N&M", 1, note="<i>")],
    )
    html = render_html(result)
    assert "A&lt;b&gt;" in html and "&lt;script&gt;x" in html
    assert "S&amp;P" in html and "N&amp;M" in html and "&lt;i&gt;" in html
    assert "<script>" not in html


def test_the_new_sections_sit_between_quiet_and_failed_and_notes_come_last() -> None:
    result = ScanResult(
        quiet_sources=[("Q", 40)],
        shrunk_sources=[("S", 1, 40.0)],
        truncated_sources=[("C", 5, 9, "stopped")],
        reports=[
            SourceReport("lever", "dead", "Dead", error="HTTP 500"),
            SourceReport("workday", "acme", "Acme", 9, note="a note"),
        ],
    )
    headings = (
        "Sources that went quiet",
        "Sources that shrank",
        "Sources that were cut short",
        "Sources that failed this run",
        "Notes",
    )
    for text in (render_markdown(result), render_html(result)):
        at = [text.index(h) for h in headings]
        assert at == sorted(at), text


def test_a_clean_run_shows_none_of_them() -> None:
    text = render_markdown(ScanResult(reports=[SourceReport("lever", "a", "A", 3)]))
    assert "cut short" not in text and "shrank" not in text and "Notes" not in text
