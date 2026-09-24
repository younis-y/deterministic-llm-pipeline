"""End-to-end: the real CLI, the real pipeline, a mocked network.

Proves the wiring holds together, not just the units: config load, source
dispatch, scoring, store, digest file, and exit codes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
import respx
from typer.testing import CliRunner

from conftest import plain
from rolescan.cli import app
from rolescan.store import Store

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


def _project(tmp_path: Path, config_text: str = CONFIG) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(config_text)
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


def test_mark_rejects_an_unknown_state(tmp_path: Path) -> None:
    result = runner.invoke(app, ["mark", "https://x/1", "maybe"])
    assert result.exit_code == 2
    assert "shortlist" in plain(result.output)


def test_mark_records_a_state_readable_back(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    result = runner.invoke(app, ["mark", "https://x/1", "shortlist", "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "shortlist: https://x/1" in plain(result.output)

    async def read() -> str | None:
        async with Store(tmp_path / "seen.db") as store:
            return await store.application_state("https://x/1")

    assert asyncio.run(read()) == "shortlist"


def test_mark_twice_updates_rather_than_duplicates(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    runner.invoke(app, ["mark", "https://x/1", "shortlist", "-c", str(cfg)])
    result = runner.invoke(app, ["mark", "https://x/1", "applied", "-c", str(cfg)])
    assert result.exit_code == 0, result.output

    async def read() -> tuple[str | None, int]:
        async with Store(tmp_path / "seen.db") as store:
            state = await store.application_state("https://x/1")
            rows = list(
                await store.db.execute_fetchall(
                    "SELECT COUNT(*) FROM applications WHERE url = ?", ("https://x/1",)
                )
            )
            return state, int(rows[0][0])

    state, count = asyncio.run(read())
    assert state == "applied"
    assert count == 1


# --- email: three outcomes, three distinguishable results ------------------

EMAIL_CONFIG = CONFIG.replace(
    "output:\n  dir: digests\n  db_path: seen.db\n",
    "output:\n"
    "  dir: digests\n"
    "  db_path: seen.db\n"
    "  email:\n"
    "    enabled: true\n"
    "    smtp_host: smtp.example.test\n"
    "    username: me@example.test\n"
    "    password: not-a-real-password\n"
    "    to: me@example.test\n",
)


class _FakeSMTP:
    """Stands in for smtplib.SMTP. Records rather than sends."""

    sent: ClassVar[list[str]] = []

    def __init__(self, host: str, port: int, timeout: int = 30) -> None:
        self.host = host

    def __enter__(self) -> _FakeSMTP:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def starttls(self) -> None:
        return None

    def login(self, user: str, password: str) -> None:
        return None

    def send_message(self, msg: object) -> None:
        _FakeSMTP.sent.append(str(msg))


@respx.mock
def test_no_email_says_it_was_skipped(tmp_path: Path) -> None:
    """Silence after a scan used to mean either "sent" or "not configured"."""
    cfg = _project(tmp_path)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code == 0, result.output
    assert "email skipped: --no-email" in plain(result.output)


@respx.mock
def test_a_disabled_emailer_says_why_it_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EmailConfig switches itself off when the password is missing. The run
    then finished, exited 0, and said nothing at all — so the first sign was
    the 06:30 email not arriving."""
    monkeypatch.delenv("ROLESCAN_SMTP_PASS", raising=False)
    monkeypatch.delenv("JOBSCAN_SMTP_PASS", raising=False)
    no_password = EMAIL_CONFIG.replace("    password: not-a-real-password\n", "")
    cfg = _project(tmp_path, no_password)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert "email skipped" in out
    assert "ROLESCAN_SMTP_PASS" in out


@respx.mock
def test_a_missing_smtp_host_is_named_rather_than_blamed_on_the_password(
    tmp_path: Path,
) -> None:
    no_host = EMAIL_CONFIG.replace("    smtp_host: smtp.example.test\n", "")
    cfg = _project(tmp_path, no_host)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "smtp_host" in plain(result.output)


@respx.mock
def test_a_working_send_reports_success_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rolescan.digest.smtplib.SMTP", _FakeSMTP)
    _FakeSMTP.sent.clear()
    cfg = _project(tmp_path, EMAIL_CONFIG)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "emailed" in plain(result.output)
    assert len(_FakeSMTP.sent) == 1


@respx.mock
def test_an_smtp_failure_exits_non_zero_and_leaves_the_digest_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec: "SMTP failure writes the digest to disk and exits non-zero,
    so launchd records it." The disk write was there; the exit code was not,
    so a scan that never delivered anything looked, to launchd and to the
    reader, exactly like one that did."""

    def boom(host: str, port: int, timeout: int = 30) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr("rolescan.digest.smtplib.SMTP", boom)
    cfg = _project(tmp_path, EMAIL_CONFIG)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg)])
    assert result.exit_code == 1, result.output
    assert "email failed" in plain(result.output)
    assert (tmp_path / "digests" / "latest.md").is_file()


# --- the mark command --------------------------------------------------------


@respx.mock
def test_mark_fills_in_company_and_title_from_the_posting_cache(
    tmp_path: Path,
) -> None:
    """Marking a url the store already knows must not write a blank row: the
    digest's Shortlist section then has nothing to show but the url."""
    from rolescan.models import Job
    from rolescan.store import Store

    cfg = _project(tmp_path)
    job = Job(
        source="greenhouse",
        company="Acme Energy",
        title="Graduate Data Scientist",
        location="London",
        url="https://boards.greenhouse.io/acme/jobs/1",
        description="Python, energy.",
    )

    async def seed() -> None:
        async with Store(tmp_path / "seen.db") as store:
            await store.put_posting(job.url, "2026-09-20", job)

    asyncio.run(seed())

    result = runner.invoke(app, ["mark", job.url, "shortlist", "-c", str(cfg)])
    assert result.exit_code == 0, result.output

    async def read() -> list[tuple[str, str, str]]:
        async with Store(tmp_path / "seen.db") as store:
            return await store.shortlist()

    rows = asyncio.run(read())
    assert rows == [(job.url, "Acme Energy", "Graduate Data Scientist")]


@respx.mock
def test_the_digest_prints_a_mark_command_that_works_from_anywhere(
    tmp_path: Path,
) -> None:
    """`rolescan mark` defaults to ./config.yaml, so the command the digest
    printed exited 2 with "No config at config.yaml" everywhere but the
    project directory — and under launchd the config is at an absolute path."""
    cfg = _project(tmp_path)
    runner.invoke(app, ["mark", "https://x/1", "shortlist", "-c", str(cfg)])
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code == 0, result.output

    digest = (tmp_path / "digests" / "latest.md").read_text()
    assert f"rolescan mark https://x/1 applied --config {cfg}" in digest
    assert f"rolescan mark https://x/1 dismissed --config {cfg}" in digest
