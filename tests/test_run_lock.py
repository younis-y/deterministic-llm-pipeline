"""One scan at a time per store (2.5.8).

Found 2026-10-07: nothing stopped two scans on one store. Both listed the same
role and both paid for its LLM call; a second writer on a busy store failed
later with "database is locked". `rolescan scan` now holds `<db>.lock` for its
whole run, and a second one stops at once, says who holds it, and exits 4."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import plain
from rolescan.cli import app
from rolescan.storefile import RunLockedError, run_lock

CONFIG = """
profile:
  keywords: {energy: 6, python: 4}
  min_keyword_score: 4
llm:
  enabled: false
sources:
  - {kind: greenhouse, slug: acme, label: Acme Energy}
output:
  dir: digests
  db_path: seen.db
"""


def test_a_second_run_is_refused_and_told_who_holds_the_lock(tmp_path: Path) -> None:
    db = tmp_path / "seen.db"
    with (
        run_lock(db),
        pytest.raises(RunLockedError, match=f"pid {os.getpid()}"),
        run_lock(db),
    ):
        pass


def test_the_lock_is_free_again_once_the_holder_finishes(tmp_path: Path) -> None:
    db = tmp_path / "seen.db"
    with run_lock(db):
        pass
    with run_lock(db):
        pass


def test_the_lock_is_released_when_the_run_raises(tmp_path: Path) -> None:
    db = tmp_path / "seen.db"
    with pytest.raises(ValueError, match="boom"), run_lock(db):
        raise ValueError("boom")
    with run_lock(db):
        pass


def test_a_scan_while_another_run_holds_the_lock_exits_4(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG)
    with run_lock(tmp_path / "seen.db"):
        result = CliRunner().invoke(app, ["scan", "-c", str(config), "--no-email"])

    assert result.exit_code == 4
    assert "another rolescan run is using" in " ".join(plain(result.output).split())
    assert not (tmp_path / "digests").exists(), "nothing ran"
