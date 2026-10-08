"""A checked daily copy of the store (2.5.8).

Found 2026-10-07: the only copy of `seen` - the record of what has been
listed, which nothing can rebuild - was the live file, and a `cp` of a
WAL-mode file is not a copy (100 of 150 committed rows, the rest still in
`-wal`). `rolescan scan` now takes the day's copy with SQLite's backup API
before it opens the store, checks it with `integrity_check`, and keeps
`output.backup_keep` of them; `rolescan backup` takes one on demand."""

from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from conftest import plain
from rolescan.cli import app
from rolescan.storefile import BackupError, backup

CONFIG = """
profile:
  keywords: {energy: 6, python: 4}
  min_keyword_score: 4
  min_report_score: 0
llm:
  enabled: false
sources:
  - {kind: greenhouse, slug: acme, label: Acme Energy}
output:
  dir: digests
  db_path: seen.db
"""
BOARD = {
    "jobs": [
        {
            "id": 1,
            "title": "Energy Data Analyst",
            "location": {"name": "London, UK"},
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
            "content": "<p>Python and energy markets.</p>",
            "updated_at": "2026-08-20T10:00:00Z",
        }
    ]
}


def _make_db(path: Path, rows: int = 3) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE seen (uid TEXT PRIMARY KEY, title TEXT)")
        conn.executemany(
            "INSERT INTO seen VALUES (?, ?)",
            [(f"u{i}", "Analyst " * 20) for i in range(rows)],
        )
        conn.commit()


def _rows(path: Path) -> int:
    with closing(sqlite3.connect(path)) as conn:
        return int(conn.execute("SELECT count(*) FROM seen").fetchone()[0])


def test_backup_copies_rows_still_in_the_write_ahead_log(tmp_path: Path) -> None:
    """The writer stays open, so its rows are still in `-wal`."""
    db = tmp_path / "seen.db"
    _make_db(db, rows=1)
    with closing(sqlite3.connect(db)) as writer:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.executemany(
            "INSERT INTO seen VALUES (?, 'x')", [(f"w{i}",) for i in range(50)]
        )
        writer.commit()
        copy = backup(db, today=date(2026, 10, 8))

    assert copy == tmp_path / "backups" / "seen-2026-10-08.db"
    assert _rows(copy) == 51
    assert not list((tmp_path / "backups").glob("*-wal")), "one self-contained file"


def test_backup_runs_once_a_day_unless_forced(tmp_path: Path) -> None:
    db = tmp_path / "seen.db"
    _make_db(db, rows=1)
    first = backup(db, today=date(2026, 10, 8))
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("INSERT INTO seen VALUES ('late', 'x')")
        conn.commit()

    assert backup(db, today=date(2026, 10, 8)) == first
    assert first is not None and _rows(first) == 1
    assert backup(db, today=date(2026, 10, 8), force=True) == first
    assert _rows(first) == 2


def test_backup_keeps_the_newest_dailies_only(tmp_path: Path) -> None:
    db = tmp_path / "rolescan.sqlite3"
    _make_db(db)
    for day in range(1, 10):
        backup(db, keep=7, today=date(2026, 10, day))

    kept = sorted(p.name for p in (tmp_path / "backups").iterdir())
    assert kept == [f"rolescan-2026-10-0{d}.sqlite3" for d in range(3, 10)]


def test_rotation_never_deletes_a_file_it_did_not_make(tmp_path: Path) -> None:
    db = tmp_path / "seen.db"
    _make_db(db)
    folder = tmp_path / "backups"
    folder.mkdir()
    mine = (
        "seen-before-upgrade.db",
        "notes.txt",
        "other-2026-01-01.db",
        # Dated like a daily, but not one: a longer name, a different suffix.
        "seen-2026-10-01.db.keep",
        "seen-2026-10-01.sqlite3",
    )
    for name in mine:
        (folder / name).write_text("mine")

    for day in range(1, 4):
        backup(db, keep=1, today=date(2026, 10, day))

    assert sorted(p.name for p in folder.iterdir()) == sorted(
        (*mine, "seen-2026-10-03.db")
    )


def test_a_store_in_a_folder_with_a_space_and_an_accent_backs_up(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "Mobile Documents" / "Recherche d'emploi é"
    folder.mkdir(parents=True)
    _make_db(folder / "seen.db")

    copy = backup(folder / "seen.db", today=date(2026, 10, 8))

    assert copy == folder / "backups" / "seen-2026-10-08.db"
    assert _rows(copy) == 3


@pytest.mark.skipif(sys.platform == "win32", reason="? is not valid in a Windows name")
def test_a_store_in_a_folder_with_a_hash_and_a_question_mark_backs_up(
    tmp_path: Path,
) -> None:
    """Both would end a naive `file:<path>?mode=ro` URI early."""
    folder = tmp_path / "jobs #2? (old)"
    folder.mkdir()
    _make_db(folder / "seen.db")

    copy = backup(folder / "seen.db", today=date(2026, 10, 8))

    assert copy == folder / "backups" / "seen-2026-10-08.db"
    assert _rows(copy) == 3


def test_keep_zero_never_deletes_an_on_demand_copy(tmp_path: Path) -> None:
    db = tmp_path / "seen.db"
    _make_db(db)

    for day in (1, 2, 3):
        backup(db, keep=0, force=True, today=date(2026, 10, day))

    assert sorted(p.name for p in (tmp_path / "backups").iterdir()) == [
        "seen-2026-10-01.db",
        "seen-2026-10-02.db",
        "seen-2026-10-03.db",
    ]


def test_the_copy_just_made_survives_rotation_when_the_clock_went_back(
    tmp_path: Path,
) -> None:
    db = tmp_path / "seen.db"
    _make_db(db)
    for day in range(5, 12):
        backup(db, keep=7, today=date(2026, 10, day))

    copy = backup(db, keep=7, today=date(2026, 10, 1))

    assert copy is not None and copy.is_file() and _rows(copy) == 3
    assert sorted(p.name for p in (tmp_path / "backups").iterdir()) == [
        f"seen-2026-10-{day:02d}.db" for day in (1, 6, 7, 8, 9, 10, 11)
    ]


def test_the_copy_is_written_under_a_name_no_other_process_shares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manual `rolescan backup` during a scan's copy must not clobber it."""
    db = tmp_path / "seen.db"
    _make_db(db)
    folder = tmp_path / "backups"
    folder.mkdir()
    other = folder / "seen-2026-10-08.db.99999.tmp"
    other.write_text("another process, half way through")
    moved: list[str] = []
    real_replace = Path.replace

    def spy(self: Path, target: Path) -> Path:
        moved.append(self.name)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", spy)

    backup(db, today=date(2026, 10, 8))

    assert moved == [f"seen-2026-10-08.db.{os.getpid()}.tmp"]
    assert other.read_text() == "another process, half way through"


def test_a_folder_that_cannot_be_made_is_a_backup_error_not_a_traceback(
    tmp_path: Path,
) -> None:
    db = tmp_path / "seen.db"
    _make_db(db)
    (tmp_path / "backups").write_text("a file where the folder should be")

    with pytest.raises(BackupError, match=str(db)):
        backup(db, today=date(2026, 10, 8))


def test_a_copy_that_cannot_be_rotated_out_is_a_backup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "seen.db"
    _make_db(db)
    backup(db, keep=1, today=date(2026, 10, 1))
    real_unlink = Path.unlink

    def refuse(self: Path, missing_ok: bool = False) -> None:
        if self.suffix == ".db":
            raise PermissionError(13, "Permission denied", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", refuse)

    with pytest.raises(BackupError, match="Permission denied"):
        backup(db, keep=1, today=date(2026, 10, 2))


def test_a_damaged_store_fails_the_backup_and_keeps_the_last_good_copy(
    tmp_path: Path,
) -> None:
    db = tmp_path / "seen.db"
    _make_db(db, rows=400)
    good = backup(db, today=date(2026, 10, 7))
    with db.open("r+b") as f:  # overwrite part of a b-tree page
        f.seek(4096 * 3 + 200)
        f.write(b"\xff" * 64)

    with pytest.raises(BackupError, match=str(db)):
        backup(db, today=date(2026, 10, 8))

    assert sorted(p.name for p in (tmp_path / "backups").iterdir()) == [
        "seen-2026-10-07.db"
    ]
    assert good is not None and _rows(good) == 400


def test_no_store_yet_means_nothing_to_back_up(tmp_path: Path) -> None:
    assert backup(tmp_path / "seen.db") is None
    assert not (tmp_path / "backups").exists()


@respx.mock
def test_scan_takes_the_days_backup_before_it_writes(tmp_path: Path) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG)
    runner = CliRunner()
    first = runner.invoke(app, ["scan", "-c", str(config), "--no-email"])
    assert first.exit_code == 0, first.output
    assert not (tmp_path / "backups").exists(), "no store existed before run 1"

    second = runner.invoke(app, ["scan", "-c", str(config), "--no-email"])

    assert second.exit_code == 0, second.output
    [copy] = (tmp_path / "backups").iterdir()
    assert _rows(copy) == 1


@respx.mock
def test_the_first_scan_into_a_folder_that_does_not_exist_yet(tmp_path: Path) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json={"jobs": []})
    )
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG.replace("db_path: seen.db", "db_path: data/seen.db"))

    result = CliRunner().invoke(app, ["scan", "-c", str(config), "--no-email"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "seen.db").is_file()
    assert (tmp_path / "data" / "seen.db.lock").is_file()
    assert not (tmp_path / "data" / "backups").exists(), "nothing to back up yet"


def test_backup_command_reports_where_the_copy_went(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG + "  backup_keep: 0\n")
    _make_db(tmp_path / "seen.db")

    result = CliRunner().invoke(app, ["backup", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert "integrity_check ok" in " ".join(plain(result.output).split())
    assert len(list((tmp_path / "backups").iterdir())) == 1


def test_a_scan_over_a_damaged_store_stops_before_it_scans(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG)
    db = tmp_path / "seen.db"
    _make_db(db, rows=400)
    with db.open("r+b") as f:
        f.seek(4096 * 3 + 200)
        f.write(b"\xff" * 64)

    result = CliRunner().invoke(app, ["scan", "-c", str(config), "--no-email"])

    assert result.exit_code == 1
    assert "Nothing was scanned" in " ".join(plain(result.output).split())
    assert not (tmp_path / "digests").exists(), "nothing ran"


@respx.mock
def test_backup_keep_zero_skips_the_scans_automatic_copy(tmp_path: Path) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=BOARD)
    )
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG + "  backup_keep: 0\n")
    runner = CliRunner()

    for _ in range(2):  # the second run has a store to copy
        result = runner.invoke(app, ["scan", "-c", str(config), "--no-email"])
        assert result.exit_code == 0, result.output

    assert (tmp_path / "seen.db").is_file()
    assert not (tmp_path / "backups").exists()


def test_backup_command_with_keep_zero_leaves_earlier_copies_alone(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG + "  backup_keep: 0\n")
    db = tmp_path / "seen.db"
    _make_db(db)
    older = backup(db, keep=0, today=date(2020, 1, 1))
    assert older is not None

    result = CliRunner().invoke(app, ["backup", "-c", str(config)])

    assert result.exit_code == 0, result.output
    assert older.is_file()
    assert len(list((tmp_path / "backups").iterdir())) == 2
