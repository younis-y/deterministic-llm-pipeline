"""End-to-end: the real CLI, the real pipeline, a mocked network.

Proves the wiring holds together, not just the units: config load, source
dispatch, scoring, store, digest file, and exit codes.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from conftest import plain
from rolescan.cli import app

runner = CliRunner()

CONFIG = """
profile:
  name: Test
  summary: An energy data candidate.
  cv_dir: cvs
  locations: [london]
  keywords: {energy: 6, data scientist: 7, python: 4, trading: 6, graduate: 4}
  blockers: {uae national: 40}
  min_keyword_score: 18
  min_report_score: 55
llm:
  enabled: false
output:
  dir: digests
  db_path: seen.db
sources:
  - {kind: greenhouse, slug: acme, label: Acme Energy}
"""

BOARD = {
    "jobs": [
        {
            "id": 1,
            "title": "Graduate Data Scientist, Energy Trading",
            "location": {"name": "London, UK"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
            "content": "<p>Python, energy, trading, forecasting.</p>",
            "updated_at": "2026-08-20T10:00:00Z",
        },
        {
            "id": 2,
            "title": "Warehouse Operative",
            "location": {"name": "Leeds, UK"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/2",
            "content": "<p>Lifting boxes.</p>",
            "updated_at": "2026-08-20T10:00:00Z",
        },
    ]
}


def _project(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG)
    cvs = tmp_path / "cvs"
    cvs.mkdir()
    (cvs / "CV_EnergySystems-Modelling.tex").write_text(
        r"\begin{document}\section{Skills} Python, forecasting.\end{document}"
    )
    return cfg


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("scan", "discover", "sources", "cvs", "show", "stats", "prune"):
        assert command in plain(result.output)


def test_missing_config_exits_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", "-c", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2
    assert "No config" in plain(result.output)


@respx.mock
def test_full_scan_writes_a_digest(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )

    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code == 0, result.output

    digest = tmp_path / "digests" / "latest.md"
    assert digest.is_file()
    text = digest.read_text()
    assert "Graduate Data Scientist, Energy Trading" in text
    assert "Warehouse Operative" not in text, "the prefilter should drop this"
    assert (tmp_path / "seen.db").is_file()

    # show reprints what scan wrote
    shown = runner.invoke(app, ["show", "-c", str(cfg)])
    assert shown.exit_code == 0
    assert "Graduate Data Scientist" in shown.output

    # a second scan finds nothing new
    again = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert again.exit_code == 0
    assert "Nothing new" in digest.read_text()


@respx.mock
def test_paths_resolve_against_the_config_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cron job runs from elsewhere; the database must not follow the cwd."""
    cfg = _project(tmp_path)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code == 0, result.output
    assert not (elsewhere / "seen.db").exists()
    assert not (elsewhere / "digests").exists()
    assert (tmp_path / "seen.db").is_file()


def test_cvs_command_reads_variants(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    result = runner.invoke(app, ["cvs", "-c", str(cfg)])
    assert result.exit_code == 0
    assert "CV_EnergySystems-Modelling" in plain(result.output)


def test_stats_reports_zero_on_a_fresh_store(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    result = runner.invoke(app, ["stats", "-c", str(cfg)])
    assert result.exit_code == 0
    assert "0 postings" in plain(result.output)


@respx.mock
def test_discover_marks_good_and_bad_slugs(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(404)
    )
    result = runner.invoke(app, ["discover", "-c", str(cfg)])
    assert result.exit_code == 0
    assert "FAIL" in plain(result.output)
    assert "0 verified" in plain(result.output)
    assert "1 broken" in plain(result.output)
    assert "boards.greenhouse.io" in plain(result.output), "should print the slug hint"


@respx.mock
def test_discover_never_reports_an_unverifiable_slug_as_working(
    tmp_path: Path,
) -> None:
    """End-to-end guard on the false-OK bug, through the real CLI."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        CONFIG.replace(
            "  - {kind: greenhouse, slug: acme, label: Acme Energy}",
            "  - {kind: smartrecruiters, slug: Ghost, label: Ghost Corp}\n"
            "  - {kind: greenhouse, slug: acme, label: Acme Energy}\n"
            "  - {kind: adzuna, slug: gb, label: Adzuna UK, queries: [energy]}",
        )
    )
    (tmp_path / "cvs").mkdir()
    respx.get("https://api.smartrecruiters.com/v1/companies/Ghost/postings").mock(
        return_value=httpx.Response(200, json={"totalFound": 0, "content": []})
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )

    result = runner.invoke(app, ["discover", "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "UNKNOWN" in plain(result.output), "the ghost slug must not read as OK"
    assert "SKIPPED" in plain(result.output), "keyless Adzuna must not read as OK"
    assert "1 verified" in plain(result.output)
    assert "1 unverifiable" in plain(result.output)
    assert "1 skipped" in plain(result.output)


@respx.mock
def test_discover_marks_disabled_sources(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        CONFIG.replace(
            "  - {kind: greenhouse, slug: acme, label: Acme Energy}",
            "  - {kind: greenhouse, slug: acme, label: Acme, enabled: false}",
        )
    )
    (tmp_path / "cvs").mkdir()
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    result = runner.invoke(app, ["discover", "-c", str(cfg)])
    assert result.exit_code == 0
    assert "disabled" in plain(result.output), "probing a disabled source must say so"
