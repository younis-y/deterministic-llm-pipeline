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

from conftest import OLLAMA_MODEL, mock_ollama, plain
from rolescan.cli import app
from rolescan.models import Job, ScoredJob
from rolescan.store import Store

runner = CliRunner()

CONFIG = """
profile:
  name: Test
  summary: An energy data candidate.
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
    return cfg


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("scan", "discover", "sources", "show", "stats", "prune"):
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


async def _seen_count(db: Path) -> int:
    async with Store(db) as store:
        return await store.count()


@respx.mock
def test_seen_is_written_only_after_the_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write_digest` fails, so nothing may be recorded and the exit code is
    non-zero. Before 2.5.7 `seen` was committed inside `run_scan`, before the
    digest existed, so a crash between the two lost the run for good."""
    import rolescan.cli as cli_module

    cfg = _project(tmp_path)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )

    def boom(*a: object, **k: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(cli_module, "write_digest", boom)
    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code != 0
    assert asyncio.run(_seen_count(tmp_path / "seen.db")) == 0


@respx.mock
def test_seen_is_written_even_if_terminal_rendering_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest is on disk, so its postings must be recorded before anything
    is drawn to the terminal. The record used to come after the rendering, so
    an exception there left a digest whose postings were reported again on the
    next run. The exit code is still non-zero: the command did fail."""
    import rolescan.cli as cli_module

    cfg = _project(tmp_path)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    real_write = cli_module.write_digest
    real_print = cli_module.console.print
    written: list[Path] = []

    def write_then_arm(*a: object, **k: object) -> Path:
        path = real_write(*a, **k)  # type: ignore[arg-type]
        written.append(path)
        return path

    def print_or_fail(*a: object, **k: object) -> None:
        if written:
            raise RuntimeError("terminal rendering failed")
        real_print(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(cli_module, "write_digest", write_then_arm)
    monkeypatch.setattr(cli_module.console, "print", print_or_fail)

    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])

    assert result.exit_code != 0
    assert written and written[0].is_file(), "the digest was written first"
    assert asyncio.run(_seen_count(tmp_path / "seen.db")) > 0


@respx.mock
def test_a_first_dry_run_records_nothing_and_leaves_no_latest(
    tmp_path: Path,
) -> None:
    """The guard `test_dry_run_writes_its_own_file` cannot be: with a fresh
    store every posting is new, so a dry run that recorded would show up as
    rows in `seen`, and one that wrote `latest.md` would leave the file."""
    cfg = _project(tmp_path)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    dry = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email", "--dry"])
    assert dry.exit_code == 0, dry.output
    assert asyncio.run(_seen_count(tmp_path / "seen.db")) == 0
    assert (tmp_path / "digests" / "digest-dry.md").is_file()
    assert not (tmp_path / "digests" / "latest.md").exists()


@respx.mock
def test_dry_run_writes_its_own_file(tmp_path: Path) -> None:
    """A dry run after a real run leaves `latest.md` as the real run's digest
    and writes `digest-dry.md` beside it."""
    cfg = _project(tmp_path)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    real = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert real.exit_code == 0, real.output
    latest_before = (tmp_path / "digests" / "latest.md").read_text()
    dry = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email", "--dry"])
    assert dry.exit_code == 0, dry.output
    assert (tmp_path / "digests" / "digest-dry.md").is_file()
    assert (tmp_path / "digests" / "latest.md").read_text() == latest_before


def test_unsee_command_removes_a_posting(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    job = Job(
        source="greenhouse:acme",
        company="Acme",
        title="Energy Analyst",
        url="https://boards.greenhouse.io/acme/jobs/9",
        description="energy",
    )

    async def seed() -> None:
        async with Store(tmp_path / "seen.db") as store:
            await store.record_all([(ScoredJob(job=job, keyword_score=20), "judged")])

    asyncio.run(seed())
    result = runner.invoke(app, ["unsee", job.url, "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "1 row removed" in plain(result.output)
    assert asyncio.run(_seen_count(tmp_path / "seen.db")) == 0


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


# --- a slow preflight must not make the CLI contradict the digest ----------

OLLAMA_CONFIG = CONFIG.replace(
    "llm:\n  enabled: false",
    f"llm:\n  enabled: true\n  backend: ollama\n  model: {OLLAMA_MODEL}\n  mode: judge",
)


@respx.mock
def test_a_failed_preflight_does_not_claim_nothing_was_scored(tmp_path: Path) -> None:
    """The preflight is a few seconds by design and `FitScorer` is built
    regardless of it, so a slow `/api/tags` left the CLI printing "LLM
    scoring did not run" in bold red over postings that carry fit scores.
    Keeping the short timeout is right; claiming the scan did not happen is
    not."""
    cfg = _project(tmp_path, OLLAMA_CONFIG)
    mock_ollama(model=OLLAMA_MODEL)
    # The probe times out; the chat endpoint mocked above still answers.
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ReadTimeout("timed out")
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )

    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert "LLM scoring did not run" not in out
    assert "scoring ran anyway" in out

    text = (tmp_path / "digests" / "latest.md").read_text()
    assert "did not run at all" not in text
    assert "scoring ran anyway" in text
    assert "confidence" in text, "the postings really do carry LLM scores"


ENRICHER_CONFIG = OLLAMA_CONFIG.replace(
    f"model: {OLLAMA_MODEL}\n  mode: judge",
    f"model: {OLLAMA_MODEL}\n  mode: judge\n  enricher: not-a-real-enricher",
)


@respx.mock
def test_an_unknown_enricher_does_not_claim_the_backend_failed(tmp_path: Path) -> None:
    """Round 2 of this task's review: a misspelt `llm.enricher` on an
    otherwise healthy backend must print its own note, never the "backend
    check failed" / "did not run at all" language those exist for."""
    cfg = _project(tmp_path, ENRICHER_CONFIG)
    mock_ollama(model=OLLAMA_MODEL)
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )

    result = runner.invoke(app, ["scan", "-c", str(cfg), "--no-email"])
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert "not-a-real-enricher" in out
    assert "LLM scoring did not run" not in out
    assert "backend check failed" not in out

    text = (tmp_path / "digests" / "latest.md").read_text()
    assert "not-a-real-enricher" in text
    assert "did not run at all" not in text
    assert "pre-scan backend check failed" not in text
