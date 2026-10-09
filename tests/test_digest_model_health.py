"""The digest's "Model health" line (2.6.0): an alarm in the opening block, in
both renderers, shown even when it is the only thing to report, and absent
when no rate is more than twice its median."""

from __future__ import annotations

from rolescan.digest import render_html, render_markdown
from rolescan.health import HealthFlag
from rolescan.models import FitVerdict, Job, ScoredJob, Verdict
from rolescan.pipeline import ScanResult

FLAGS = [
    HealthFlag("level overridden", 14, 20, 0.3),
    HealthFlag("quotes rejected", 5, 20, 0.0),
]


def _result(**kw: object) -> ScanResult:
    return ScanResult(model_health=list(FLAGS), **kw)  # type: ignore[arg-type]


def _role() -> ScoredJob:
    job = Job(
        source="t",
        company="Acme",
        title="Graduate Analyst",
        url="https://acme.example/1",
        description="Analysis.",
    )
    fit = FitVerdict(
        fit_score=80, verdict=Verdict.APPLY, confidence="high", reason="Fits."
    )
    return ScoredJob(job=job, keyword_score=30, fit=fit)


def test_markdown_names_each_rate_with_its_median_when_it_is_the_only_alarm() -> None:
    text = render_markdown(_result())

    assert "## Needs attention" in text
    assert "**Model health**" in text
    assert "level overridden 70% (median 30%)" in text
    assert "quotes rejected 25% (median 0%)" in text
    assert "20 postings" in text
    assert "the last five runs" in text


def test_the_line_opens_the_digest_ahead_of_the_roles() -> None:
    text = render_markdown(_result(reportable=[_role()]))

    assert text.index("Model health") < text.index("Worth a look")
    assert text.index("Needs attention") < text.index("Model health")


def test_the_line_sits_in_the_block_with_the_other_alarms() -> None:
    text = render_markdown(
        _result(quiet_sources=[("Acme", 40)], llm_errors=2, llm_error_detail="x")
    )

    block = text.split("---")[0]
    assert "Model health" in block and "went quiet" in block


def test_html_has_the_same_line_and_alarm_block() -> None:
    html = render_html(_result())

    assert "Needs attention" in html
    assert "Model health" in html
    assert "level overridden 70% (median 30%)" in html
    assert "quotes rejected 25% (median 0%)" in html


def test_html_line_opens_the_digest_ahead_of_the_roles() -> None:
    html = render_html(_result(reportable=[_role()]))

    assert html.index("Model health") < html.index("Graduate Analyst")


def test_a_run_with_no_flags_says_nothing_about_the_model() -> None:
    result = ScanResult()

    assert "Model health" not in render_markdown(result)
    assert "Model health" not in render_html(result)
    assert "Needs attention" not in render_markdown(result)


def test_the_line_names_the_real_number_of_runs_in_the_baseline() -> None:
    """Three earlier runs are enough to compare with, and then the median is
    of three, not five."""
    for runs, words in ((3, "three"), (4, "four"), (5, "five")):
        flags = [HealthFlag("level overridden", 14, 20, 0.3, runs)]
        result = ScanResult(model_health=flags)

        assert f"the last {words} runs" in render_markdown(result)
        assert f"the last {words} runs" in render_html(result)
        if runs != 5:
            assert "five" not in render_markdown(result)
