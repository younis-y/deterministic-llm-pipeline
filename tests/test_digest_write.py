"""2.5.7: the digest is the record of a run, so it must exist before anything
is marked seen (storage audit S2: a crash between the two lost the run for
good), one scan must not overwrite another's file (S3: 28 Sep 2026 had four
scans and one file), and a dry run must leave `latest.md` alone."""

from __future__ import annotations

import re
from pathlib import Path

from rolescan.digest import write_digest


def test_each_scan_gets_its_own_file(tmp_path: Path) -> None:
    a = write_digest("first", tmp_path)
    b = write_digest("second", tmp_path)
    assert a != b
    assert a.read_text() == "first"
    assert b.read_text() == "second"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{4}(?:-\d+)?\.md", a.name)
    assert (tmp_path / "latest.md").read_text() == "second"


def test_a_named_digest_does_not_touch_latest(tmp_path: Path) -> None:
    write_digest("real", tmp_path)
    path = write_digest("dry", tmp_path, name="digest-dry.md")
    assert path.name == "digest-dry.md"
    assert (tmp_path / "latest.md").read_text() == "real"
