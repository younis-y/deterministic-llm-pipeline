"""The store as a file: who may write it (2.5.8).

`Store` talks to the database through a connection. What sits outside any
connection is kept here:

  run_lock   an advisory lock next to the database, held by a scan for its
             whole run, so a second scan started while one is still running
             stops at once with a message instead of listing the same roles
             twice and paying for the same LLM calls twice.

Stdlib only (`fcntl` on POSIX, `msvcrt` on Windows).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["RunLockedError", "run_lock"]


class RunLockedError(RuntimeError):
    """Another rolescan run holds the lock on this database."""


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
