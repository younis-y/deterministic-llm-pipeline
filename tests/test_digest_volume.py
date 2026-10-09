"""The digest at volume.

The email is read on a phone and Gmail clips a message at about 102 KB. Two
things used to go wrong together: the section "Hidden by your rules" had no
length cap, at about a third of a kilobyte a line, and the alarms (a source
that went quiet, a model that stopped answering) came after it, so a long
enough list pushed the one thing the reader must see past the clip.

The order is now: alarms, roles, hidden. The hidden list is capped at
`output.hidden_max` lines, counted per rule, with the whole list in a file
beside the digest and named in it. In the email the roles spend the room first
and the hidden list gets what they leave, so no `hidden_max` can push a role
out of it."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from rolescan.cli import app
from rolescan.config import Config, OutputConfig
from rolescan.digest import (
    hidden_list_path,
    next_digest_path,
    render_hidden_list,
    render_html,
    render_markdown,
    write_digest,
)
from rolescan.models import Confidence, FitVerdict, Job, ScoredJob, Verdict
from rolescan.pipeline import ScanResult, SourceReport

runner = CliRunner()

#: Gmail clips a message past 102 KB; the digest keeps well under that.
BUDGET = 90_000


def _job(n: int, company: str = "Acme") -> Job:
    return Job(
        source="greenhouse",
        company=f"{company} {n:03d}",
        title=f"Senior Energy Data Scientist {n:03d}",
        location="London, United Kingdom",
        url=f"https://careers.example.test/jobs/{company.lower()}-{n:03d}",
        description="Python and power-market forecasting.",
    )


def _role(n: int) -> ScoredJob:
    return ScoredJob(
        job=_job(n),
        keyword_score=60,
        fit=FitVerdict(
            fit_score=90 - n % 40,
            verdict=Verdict.APPLY,
            confidence=Confidence.HIGH,
            reason=(
                "Strong overlap on power-market forecasting and Python; the "
                "advert asks for four years and the profile has them."
            ),
            blockers=[],
            keywords_missing=["kdb+", "Spark"],
        ),
    )


def _skipped(n: int, rule: str, score: int = 40) -> ScoredJob:
    return ScoredJob(
        job=_job(n, "Skip"),
        keyword_score=score,
        fit=FitVerdict(
            fit_score=score,
            verdict=Verdict.SKIP,
            confidence=Confidence.HIGH,
            reason=f'Skip: advert asks for "{n}+ years of experience"',
            blockers=[],
            keywords_missing=[],
            rule=rule,
        ),
    )


def _rejected(n: int, how: str) -> ScoredJob:
    """A prefilter reject listed under the keyword gate or the weighted terms."""
    if how == "gate":
        return ScoredJob(job=_job(n, "Near"), keyword_score=14, hidden_as="gate")
    if how == "blockers":
        return ScoredJob(
            job=_job(n, "Weighted"),
            keyword_score=10,
            keyword_penalties=["director"],
            hidden_as="blockers",
        )
    return ScoredJob(
        job=_job(n, "Termed"), keyword_score=30, blocker_hits=["security clearance"]
    )


def _failed() -> SourceReport:
    return SourceReport(kind="lever", slug="dead", label="Dead", error="HTTP 500")


def _volume_result(roles: int = 60, hidden: int = 500) -> ScanResult:
    rejects = [
        _rejected(n, ("gate", "blockers", "terms")[n % 3]) for n in range(hidden)
    ]
    return ScanResult(
        unique=2000,
        reportable=[_role(n) for n in range(roles)],
        rule_hidden=rejects,
        reports=[_failed()],
        quiet_sources=[("Acme board", 40)],
        gate=20,
    )


def _html_bytes(html: str) -> int:
    return len(html.encode("utf-8"))


# --- alarms come first ------------------------------------------------------


def test_markdown_puts_the_alarms_before_the_roles_and_the_hidden_list() -> None:
    result = _volume_result(roles=2, hidden=3)
    result.llm_errors = 2
    result.llm_error_detail = "HTTP 500"
    text = render_markdown(result)
    alarms = [
        text.index("Sources that failed"),
        text.index("Sources that went quiet"),
        text.index("LLM scoring failed"),
    ]
    assert (
        max(alarms)
        < text.index("## Worth a look")
        < text.index("## Hidden by your rules")
    )


def test_html_puts_the_alarms_before_the_roles_and_the_hidden_list() -> None:
    result = _volume_result(roles=2, hidden=3)
    result.llm_errors = 2
    result.llm_error_detail = "HTTP 500"
    html = render_html(result)
    alarms = [
        html.index("Sources that failed"),
        html.index("Sources that went quiet"),
        html.index("LLM scoring failed"),
    ]
    assert max(alarms) < html.index("Worth a look") < html.index("Hidden by your rules")


def test_the_alarms_come_before_a_nothing_to_report_line_too() -> None:
    result = _volume_result(roles=0, hidden=0)
    text = render_markdown(result)
    assert text.index("Sources that failed") < text.index("Nothing new worth")
    html = render_html(result)
    assert html.index("Sources that failed") < html.index("Nothing new worth")


def test_no_alarm_block_when_nothing_is_wrong() -> None:
    result = ScanResult(unique=3, reportable=[_role(1)])
    text = render_markdown(result)
    assert "Needs attention" not in text
    assert "Needs attention" not in render_html(result)


# --- per-rule counts --------------------------------------------------------


def test_each_rule_group_shows_its_count() -> None:
    result = ScanResult(
        rule_hidden=[
            _skipped(1, "years"),
            _skipped(2, "years"),
            _skipped(3, "level"),
        ]
    )
    text = render_markdown(result)
    assert "**Years of experience** (2)" in text
    assert "**Level** (1)" in text
    html = render_html(result)
    assert "Years of experience (2)" in html
    assert "Level (1)" in html


# --- the cap ----------------------------------------------------------------


def _capped_result() -> ScanResult:
    """Six hidden: four under `years` (scores 70, 60, 50, 40), two under
    `level` (65, 30). A cap of three keeps the three highest scores."""
    return ScanResult(
        unique=50,
        rule_hidden=[
            _skipped(1, "years", 70),
            _skipped(2, "years", 60),
            _skipped(3, "years", 50),
            _skipped(4, "years", 40),
            _skipped(5, "level", 65),
            _skipped(6, "level", 30),
        ],
    )


def test_the_hidden_section_lists_the_highest_scoring_up_to_the_cap() -> None:
    text = render_markdown(_capped_result(), hidden_max=3, hidden_file="x-hidden.md")
    section = text[text.index("## Hidden by your rules") :]
    listed = re.findall(r"\(https://careers\.example\.test/jobs/skip-(\d+)\)", section)
    assert sorted(listed) == ["001", "002", "005"]


def test_a_capped_section_says_how_many_it_left_out_and_where_they_are() -> None:
    text = render_markdown(_capped_result(), hidden_max=3, hidden_file="x-hidden.md")
    section = text[text.index("## Hidden by your rules") :]
    assert "6 hidden" in section
    assert "`x-hidden.md`" in section
    assert "highest scores" in section
    # Counts for every rule, including the group that lost rows.
    assert "Years of experience 4" in section and "Level 2" in section
    assert "**Years of experience** (2 of 4 listed)" in section
    assert "**Level** (1 of 2 listed)" in section
    # The stats line says the list is partial, with the file.
    assert "6 hidden by your rules (3 listed below, all 6 in x-hidden.md)" in text


def test_the_html_section_is_capped_the_same_way() -> None:
    html = render_html(_capped_result(), hidden_max=3, hidden_file="x-hidden.md")
    section = html[html.index("Hidden by your rules") :]
    listed = re.findall(r"careers\.example\.test/jobs/skip-(\d+)", section)
    assert sorted(set(listed)) == ["001", "002", "005"]
    assert "x-hidden.md" in section
    assert "Years of experience (2 of 4 listed)" in section
    assert "6 hidden by your rules (3 listed below, all 6 in x-hidden.md)" in html


def test_a_group_with_no_listed_row_is_in_the_counts_not_the_list() -> None:
    result = ScanResult(
        rule_hidden=[_skipped(1, "years", 70), _skipped(2, "field", 10)]
    )
    text = render_markdown(result, hidden_max=1, hidden_file="x-hidden.md")
    section = text[text.index("## Hidden by your rules") :]
    assert "Field 1" in section, "its count is in the summary"
    assert "**Field**" not in section, "and it has no group, with nothing listed"


def test_no_cap_lists_everything_as_before() -> None:
    text = render_markdown(_capped_result())
    assert len(re.findall(r"skip-\d{3}\)", text)) == 6
    assert "highest scores" not in text
    assert "(listed below)" in text


def test_a_list_within_the_cap_is_complete_and_names_no_file() -> None:
    text = render_markdown(_capped_result(), hidden_max=60, hidden_file="x-hidden.md")
    assert len(re.findall(r"skip-\d{3}\)", text)) == 6
    assert "x-hidden.md" not in text
    assert "6 hidden by your rules (listed below)" in text


def test_a_cap_of_zero_lists_nothing_but_counts_everything() -> None:
    text = render_markdown(_capped_result(), hidden_max=0, hidden_file="x-hidden.md")
    section = text[text.index("## Hidden by your rules") :]
    assert not re.findall(r"skip-\d{3}\)", section)
    assert "Years of experience 4" in section and "`x-hidden.md`" in section
    assert "6 hidden by your rules (0 listed below, all 6 in x-hidden.md)" in text


def test_a_cap_with_no_file_still_says_the_list_is_partial() -> None:
    text = render_markdown(_capped_result(), hidden_max=3)
    assert "6 hidden by your rules (3 listed below)" in text
    assert "output.hidden_max" in text


# --- the full list file -----------------------------------------------------


def test_the_full_list_has_every_row_and_the_counts() -> None:
    result = _capped_result()
    text = render_hidden_list(result)
    assert len(re.findall(r"skip-\d{3}\)", text)) == 6
    assert "**Years of experience** (4)" in text and "**Level** (2)" in text
    assert text.startswith("# Hidden by your rules")


def test_the_full_list_is_empty_when_nothing_was_hidden() -> None:
    assert render_hidden_list(ScanResult()) == ""


def test_the_list_file_sits_beside_the_digest_and_shares_its_stem() -> None:
    assert hidden_list_path(Path("d/2026-10-09T0630.md")) == Path(
        "d/2026-10-09T0630-hidden.md"
    )
    assert hidden_list_path(Path("d/digest-dry.md")) == Path("d/digest-dry-hidden.md")


# --- the digest's own path, chosen before it is rendered ---------------------


def test_next_digest_path_is_the_path_write_digest_will_use(tmp_path: Path) -> None:
    first = next_digest_path(tmp_path)
    assert not first.exists(), "choosing a path writes nothing"
    assert write_digest("a", tmp_path, path=first) == first
    assert (tmp_path / "latest.md").read_text() == "a", "a stamped digest is latest"
    second = next_digest_path(tmp_path)
    assert second != first
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{4}-2\.md", second.name)


def test_a_named_path_never_becomes_latest(tmp_path: Path) -> None:
    write_digest("real", tmp_path)
    path = next_digest_path(tmp_path, name="digest-dry.md")
    assert path == tmp_path / "digest-dry.md"
    write_digest("dry", tmp_path, name="digest-dry.md", path=path)
    assert (tmp_path / "latest.md").read_text() == "real"


# --- the size budget ---------------------------------------------------------


def test_sixty_roles_and_five_hundred_hidden_stay_under_the_email_budget() -> None:
    """60 roles at about 1.7 KB a card would be 100 KB on their own, so the
    cards give way to one line each once the budget is spent."""
    result = _volume_result(roles=60, hidden=500)
    html = render_html(result, hidden_max=60, hidden_file="digest-hidden.md")
    assert _html_bytes(html) < BUDGET

    # Nothing is lost: every one of the 60 roles is still linked.
    for n in range(60):
        assert f"jobs/acme-{n:03d}" in html
    # The first roles are still full cards, and the rest are said to be short.
    assert html.count(">Apply</a>") >= 20
    assert html.count(">Apply</a>") < 60
    assert "one line each" in html

    # The roles spent the room first, so the hidden list is what is left of
    # the 60 cap: some rows, every rule's count, and the file for the rest.
    hidden = html[html.index("Hidden by your rules") :]
    listed = re.findall(r"jobs/(?:near|weighted|termed)-\d{3}", hidden)
    assert 0 < len(listed) < 60
    assert "500 hidden" in hidden
    assert "digest-hidden.md" in hidden

    # And the alarms sit at the very top, nowhere near the clip.
    assert html.index("Sources that failed") < 4_000
    assert html.index("Sources that went quiet") < 6_000


def test_a_raised_hidden_max_never_pushes_roles_out_of_the_email() -> None:
    """The roles are laid out first and the hidden list spends what is left:
    forty full cards stay full cards whatever `hidden_max` allows, and the
    list is cut to what fits, with the rest left to the digest on disk."""
    result = _volume_result(roles=40, hidden=500)

    html = render_html(result, hidden_max=200, hidden_file="digest-hidden.md")

    assert _html_bytes(html) < BUDGET
    assert html.count(">Apply</a>") == 40, "every role is still a full card"
    assert "one line each" not in html
    hidden = html[html.index("Hidden by your rules") :]
    listed = re.findall(r"jobs/(?:near|weighted|termed)-\d{3}", hidden)
    assert 0 < len(listed) < 200, "some of the list, not all of it"
    assert "500 hidden" in hidden
    assert f"Listed below: {len(listed)} postings" in hidden
    assert "digest-hidden.md" in hidden
    assert "Gmail clips" in hidden, "and it says why the list stops there"
    assert f"{len(listed)} listed below, all 500 in digest-hidden.md" in html


def test_a_list_that_fits_after_the_roles_is_listed_whole() -> None:
    result = _volume_result(roles=3, hidden=150)

    html = render_html(result, hidden_max=200, hidden_file="digest-hidden.md")

    hidden = html[html.index("Hidden by your rules") :]
    assert len(re.findall(r"jobs/(?:near|weighted|termed)-\d{3}", hidden)) == 150
    assert "digest-hidden.md" not in html, "nothing was cut, so no file is named"
    assert "Gmail clips" not in html
    assert "150 hidden by your rules (listed below)" in html


def test_many_roles_leave_the_hidden_section_its_counts_and_a_pointer() -> None:
    """With 60 roles using the budget, the hidden section keeps what the
    reader needs to know it exists: each rule's count and where the rest is."""
    result = _volume_result(roles=60, hidden=500)

    html = render_html(result, hidden_max=200, hidden_file="digest-hidden.md")

    assert _html_bytes(html) < BUDGET
    for n in range(60):
        assert f"jobs/acme-{n:03d}" in html
    hidden = html[html.index("Hidden by your rules") :]
    assert "500 hidden" in hidden and "digest-hidden.md" in hidden


def test_a_list_cut_for_size_without_a_file_points_to_the_digest_on_disk() -> None:
    """`hidden_max` did not cut this list, so no file was written and none may
    be named: the whole of it is in the Markdown digest."""
    result = _volume_result(roles=45, hidden=300)

    html = render_html(result, hidden_max=300, hidden_file="digest-hidden.md")

    assert _html_bytes(html) < BUDGET
    hidden = html[html.index("Hidden by your rules") :]
    assert "digest-hidden.md" not in html
    assert "All 300 are in the digest on disk" in hidden
    assert "Gmail clips" in hidden


def test_the_markdown_digest_has_the_same_sixty_hidden_lines() -> None:
    result = _volume_result(roles=60, hidden=500)
    text = render_markdown(result, hidden_max=60, hidden_file="digest-hidden.md")
    section = text[text.index("## Hidden by your rules") :]
    assert len(re.findall(r"- \*\*(?:Near|Weighted|Termed) ", section)) == 60
    assert "`digest-hidden.md`" in section
    assert (
        "500 hidden by your rules (60 listed below, all 500 in digest-hidden.md)"
        in text
    )


def test_the_full_file_holds_all_five_hundred() -> None:
    result = _volume_result(roles=0, hidden=500)
    text = render_hidden_list(result)
    assert len(re.findall(r"- \*\*(?:Near|Weighted|Termed) ", text)) == 500


def test_a_role_over_the_budget_is_never_dropped_silently() -> None:
    """An absurd number of roles: the cards give way to lines, and past the
    lines the email says how many are left to the digest on disk."""
    result = ScanResult(reportable=[_role(n) for n in range(600)])
    html = render_html(result)
    assert _html_bytes(html) < BUDGET
    assert "more roles in the digest on disk" in html


def test_a_small_digest_is_unchanged_by_the_budget() -> None:
    result = ScanResult(unique=3, reportable=[_role(1), _role(2)])
    html = render_html(result)
    assert html.count(">Apply</a>") == 2
    assert "one line each" not in html


# --- config and the CLI ------------------------------------------------------


def test_hidden_max_defaults_to_sixty_and_cannot_be_negative() -> None:
    assert OutputConfig().hidden_max == 60
    with pytest.raises(ValueError, match="hidden_max"):
        OutputConfig(hidden_max=-1)


CONFIG = """
profile:
  name: Test
  summary: An energy data candidate.
  locations: [london]
  keywords: {energy: 6, python: 4}
  min_keyword_score: 40
  hidden_gate_margin: 40
  min_report_score: 60
llm:
  enabled: false
output:
  dir: digests
  db_path: seen.db
  hidden_max: 3
sources:
  - {kind: greenhouse, slug: acme, label: Acme Energy}
"""


def _board(n: int) -> dict[str, object]:
    return {
        "jobs": [
            {
                "id": i,
                "title": f"Energy Analyst {i}",
                "location": {"name": "London, UK"},
                "absolute_url": f"https://boards.example.test/acme/jobs/{i}",
                "content": "<p>Energy. Python.</p>",
                "updated_at": "2026-08-20T10:00:00Z",
            }
            for i in range(n)
        ]
    }


@respx.mock
def test_scan_writes_the_full_hidden_list_beside_the_digest(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_board(8))
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code == 0, result.output

    digests = tmp_path / "digests"
    stamped = [
        p
        for p in digests.glob("*.md")
        if p.name != "latest.md" and not p.name.endswith("-hidden.md")
    ]
    assert len(stamped) == 1
    digest = stamped[0]
    hidden_file = digests / f"{digest.stem}-hidden.md"
    assert hidden_file.is_file()

    full = hidden_file.read_text()
    assert len(re.findall(r"boards\.example\.test/acme/jobs/\d", full)) == 8
    text = digest.read_text()
    assert len(re.findall(r"boards\.example\.test/acme/jobs/\d", text)) == 3
    assert hidden_file.name in text
    assert (digests / "latest.md").read_text() == text


@respx.mock
def test_a_dry_scan_writes_its_own_hidden_list(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_board(8))
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email", "--dry"])
    assert result.exit_code == 0, result.output
    digests = tmp_path / "digests"
    assert (digests / "digest-dry-hidden.md").is_file()
    assert "digest-dry-hidden.md" in (digests / "digest-dry.md").read_text()
    assert not (digests / "latest.md").exists()


@respx.mock
def test_no_list_file_when_the_list_fits_the_cap(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG.replace("hidden_max: 3", "hidden_max: 60"))
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_board(8))
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code == 0, result.output
    assert not list((tmp_path / "digests").glob("*-hidden.md"))


def test_the_default_config_validates_with_a_hidden_max() -> None:
    cfg = Config.model_validate({"output": {"hidden_max": 10}})
    assert cfg.output.hidden_max == 10
