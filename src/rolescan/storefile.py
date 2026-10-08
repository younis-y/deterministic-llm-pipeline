"""The store as a file: who may write it, and a copy to go back to (2.5.8).

`Store` talks to the database through a connection. Two jobs sit outside any
connection and are kept here:

  run_lock   an advisory lock next to the database, held by a scan for its
             whole run, so a second scan started while one is still running
             stops at once with a message instead of listing the same roles
             twice and paying for the same LLM calls twice.
  backup     a consistent copy of the database, taken with SQLite's online
             backup API and checked with `integrity_check`, kept as one file
             per day for `keep` days. A plain `cp` of a WAL-mode file is not a
             backup: it can miss rows that are committed but still in `-wal`
             (measured 2026-10-07: 100 of 150 committed rows).

Both are stdlib only (`fcntl` on POSIX, `msvcrt` on Windows, `sqlite3`).
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

__all__ = ["BackupError", "RunLockedError", "backup", "run_lock"]


class RunLockedError(RuntimeError):
    """Another rolescan run holds the lock on this database."""


class BackupError(RuntimeError):
    """The copy could not be made, or failed its integrity check."""


def _lock_path(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.name}.lock")


def _try_lock(fd: int) -> bool:
    """Take an exclusive, non-blocking lock on `fd`; False if someone has it."""
    if sys.platform == "win32":  # pragma: no cover - not exercised on POSIX CI
        import msvcrt

        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _holder(fd: int) -> str:
    """What the lock holder wrote into the file: its pid and start time."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        text = os.read(fd, 200).decode(errors="replace").strip()
    except OSError:  # pragma: no cover - Windows refuses to read a locked byte
        text = ""
    return text or "holder unknown"


@contextmanager
def run_lock(db_path: Path) -> Iterator[None]:
    """Hold `<db>.lock` for the duration of the block.

    Raises `RunLockedError`, naming the holder's pid and start time, when
    another process already holds it. The lock belongs to the open file, so
    the operating system drops it when the holder exits, however it exits: a
    crashed scan never leaves a stale lock behind. The file itself stays, on
    purpose: deleting it on release would let a third process lock a new
    file while the second still waits on the old one.
    """
    path = _lock_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _try_lock(fd):
            msg = (
                f"another rolescan run is using {db_path} ({_holder(fd)}). "
                "Wait for it to finish, then run again."
            )
            raise RunLockedError(msg)
        started = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M")
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"pid {os.getpid()}, started {started}".encode())
        yield
    finally:
        os.close(fd)


def _backup_name(db_path: Path, day: date) -> str:
    suffix = db_path.suffix or ".sqlite3"
    return f"{db_path.stem}-{day:%Y-%m-%d}{suffix}"


def _dailies(directory: Path, db_path: Path) -> list[Path]:
    """This database's daily copies in `directory`, oldest first."""
    suffix = re.escape(db_path.suffix or ".sqlite3")
    shape = re.compile(rf"{re.escape(db_path.stem)}-\d{{4}}-\d{{2}}-\d{{2}}{suffix}")
    return sorted(p for p in directory.iterdir() if shape.fullmatch(p.name))


def backup(
    db_path: Path,
    *,
    keep: int = 7,
    force: bool = False,
    today: date | None = None,
) -> Path | None:
    """Copy `db_path` to `backups/<name>-<YYYY-MM-DD><suffix>` beside it.

    One copy per day: when today's exists already it is returned untouched,
    unless `force`. The copy is made with `sqlite3.Connection.backup` from a
    read-only connection (a consistent snapshot even while another process
    writes), switched to a rollback journal so it is a single
    self-contained file, and checked with `PRAGMA integrity_check` before it
    replaces anything. A copy that fails the check raises `BackupError` and
    is deleted: it would mean the live file is damaged, which is the moment
    to stop writing to it, not to rotate out the last good copy. Then all but
    the newest `keep` dailies are deleted (`keep` below 1 keeps one).

    Returns the copy's path, or None when there is no database yet.
    """
    if not db_path.is_file():
        return None
    day = today or datetime.now(UTC).astimezone().date()
    directory = db_path.parent / "backups"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / _backup_name(db_path, day)
    if target.exists() and not force:
        return target
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        source_uri = f"{db_path.resolve().as_uri()}?mode=ro"
        with (
            closing(sqlite3.connect(source_uri, uri=True)) as source,
            closing(sqlite3.connect(tmp)) as copy,
        ):
            source.backup(copy)
            copy.execute("PRAGMA journal_mode=DELETE")
            verdict = copy.execute("PRAGMA integrity_check").fetchone()[0]
        if verdict != "ok":
            msg = f"the copy of {db_path} failed its integrity check: {verdict}"
            raise BackupError(msg)
        tmp.replace(target)
    except (sqlite3.Error, OSError) as e:
        msg = f"could not back up {db_path}: {e}"
        raise BackupError(msg) from e
    finally:
        tmp.unlink(missing_ok=True)
    for old in _dailies(directory, db_path)[: -max(keep, 1)]:
        old.unlink()
    return target
