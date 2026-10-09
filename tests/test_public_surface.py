"""What a stranger meets first: the package metadata, the CLI banner, the
terminal table. The CV feature left the public core in 2.5.0; these pin that
nothing about it is still advertised, and that no column is left behind."""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from conftest import plain
from rolescan.cli import _ranked_table, app
from rolescan.models import Job, ScoredJob
from rolescan.pipeline import ScanResult

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "rolescan"


def _word(text: str, word: str) -> bool:
    return re.search(rf"\b{word}\b", text, re.IGNORECASE) is not None


def test_the_package_description_names_no_cv_feature() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert not _word(meta["project"]["description"], "cv")


def test_the_cli_banner_names_no_cv_feature() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert not _word(plain(result.output), "cv")


def test_there_is_no_cvs_command() -> None:
    result = CliRunner().invoke(app, ["cvs"])
    assert result.exit_code != 0


def test_no_cvs_folder_ships() -> None:
    assert not (ROOT / "cvs").exists()


def test_a_local_cvs_folder_is_ignored_by_git() -> None:
    """The folder is gone from the repository, but a user may keep their own CV
    files in one beside their config. It must never be committed by accident,
    so git ignores it."""
    try:
        done = subprocess.run(
            ["git", "check-ignore", "-q", "cvs/resume.pdf"],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout")
    if done.returncode == 128:
        pytest.skip("not a git checkout")
    assert done.returncode == 0, "cvs/ is not in .gitignore"


def test_the_ranked_table_has_a_column_for_every_value() -> None:
    """The table once declared a CV column the rows never filled."""
    job = Job(
        source="greenhouse",
        company="Acme",
        title="Data Analyst",
        location="London",
        url="https://example.test/1",
        description="Python and SQL.",
    )
    result = ScanResult(reportable=[ScoredJob(job=job, keyword_score=40)])
    table = _ranked_table(result)
    assert [column.header for column in table.columns] == ["Score", "Verdict", "Role"]
    for column in table.columns:
        assert len(list(column.cells)) == 1
    console = Console(record=True, width=100)
    console.print(table)
    assert "CV" not in console.export_text()


def test_no_module_imports_a_cv_module() -> None:
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module, *(alias.name for alias in node.names)]
            for name in names:
                assert not _word(name.replace(".", " ").replace("_", " "), "cvs?"), (
                    f"{path.name} imports {name}"
                )


def test_no_source_text_names_whose_data_a_measurement_came_from() -> None:
    """Docstrings and comments referred to the person a measurement was taken
    from, a person this project does not name. The site's owner and a product
    owner are other people and stay."""
    possessive = "owner" + "'s"
    article = "the " + "owner"
    leaks = []
    for path in sorted(SRC.rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(rf"\b{article}\b|\b{possessive}", line, re.IGNORECASE):
                leaks.append(f"{path.relative_to(ROOT)}:{number}")
    assert not leaks, leaks


def test_no_install_hint_names_a_package_that_is_not_published() -> None:
    """`pip install 'rolescan[anthropic]'` told a user to install from an index
    this project is not on. The README's form is from a checkout."""
    files = [*SRC.rglob("*.py"), ROOT / "README.md", ROOT / "config.example.yaml"]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"pip install\s+['\"]?rolescan", text), path.name
