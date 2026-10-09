"""The README's front door: `examples/quickstart.yaml` loads, and the commands
the README tells a new user to run work against it, end to end, on a mocked
board, with no key and no model."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import httpx
import respx
from typer.testing import CliRunner

from conftest import plain
from rolescan.cli import app
from rolescan.config import Config

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples" / "quickstart.yaml"
runner = CliRunner()

LEVER_BOARD: list[dict[str, Any]] = [
    {
        "id": "a1",
        "text": "Senior Data Engineer",
        "categories": {"location": "London, UK", "commitment": "Full-time"},
        "hostedUrl": "https://jobs.example.test/octoenergy/a1",
        "descriptionPlain": "Build Python pipelines for energy data.",
        "createdAt": 1790000000000,
    },
    {
        "id": "b2",
        "text": "Warehouse Operative",
        "categories": {"location": "Leeds, UK"},
        "hostedUrl": "https://jobs.example.test/octoenergy/b2",
        "descriptionPlain": "Lifting boxes.",
        "createdAt": 1790000000000,
    },
]


def _project(tmp_path: Path) -> Path:
    """What `cp examples/quickstart.yaml config.yaml` makes, in a scratch dir
    so the digest and the store land there and not beside the example."""
    cfg = tmp_path / "config.yaml"
    shutil.copy(QUICKSTART, cfg)
    return cfg


def test_the_quickstart_config_needs_no_key_and_no_model() -> None:
    cfg = Config.load(QUICKSTART)
    assert cfg.llm.enabled is False
    assert [entry.kind for entry in cfg.sources] == ["lever"]
    assert not cfg.output.email.enabled


def test_the_quickstart_claims_no_unmeasured_score_range() -> None:
    """The file once said "20 to 45 is a good match", a range nobody measured
    and the README and `config.example.yaml` do not give. Its one number is
    the one they give: about 20 to 30."""
    text = QUICKSTART.read_text(encoding="utf-8")
    assert "good match" not in text
    assert "45" not in text
    assert "20 to 30" in text


def test_the_quickstart_threshold_suits_keyword_scores() -> None:
    """The model-scale 55 would hide every keyword-only match."""
    cfg = Config.load(QUICKSTART)
    best_title_hit = 3 * sum(sorted(cfg.profile.keywords.values())[-2:])
    assert cfg.profile.min_report_score <= best_title_hit


@respx.mock
def test_the_quickstart_scan_runs_dry_and_keyword_only(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    respx.get("https://api.lever.co/v0/postings/octoenergy").mock(
        return_value=httpx.Response(200, json=LEVER_BOARD)
    )

    result = runner.invoke(
        app, ["scan", "-c", str(cfg), "--dry", "--no-llm", "--no-email"]
    )
    assert result.exit_code == 0, result.output

    digest = (tmp_path / "digests" / "digest-dry.md").read_text()
    assert "Senior Data Engineer" in digest
    assert "Warehouse Operative" not in digest, "the prefilter should drop this"
    assert not (tmp_path / "digests" / "latest.md").exists(), "--dry is not a run"

    # --dry marks nothing seen: a second dry run lists the same role again.
    again = runner.invoke(
        app, ["scan", "-c", str(cfg), "--dry", "--no-llm", "--no-email"]
    )
    assert again.exit_code == 0, again.output
    assert (
        "Senior Data Engineer" in (tmp_path / "digests" / "digest-dry.md").read_text()
    )


@respx.mock
def test_discover_accepts_the_quickstart(tmp_path: Path) -> None:
    cfg = _project(tmp_path)
    respx.get("https://api.lever.co/v0/postings/octoenergy").mock(
        return_value=httpx.Response(200, json=LEVER_BOARD)
    )
    result = runner.invoke(app, ["discover", "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "1 verified" in plain(result.output)


def _readme_quickstart_block() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```bash\n(.*?)```", section, re.DOTALL)
    assert match, "README.md has no bash block under Quickstart"
    return match.group(1)


def test_the_readme_quickstart_names_real_files_and_commands() -> None:
    block = _readme_quickstart_block()
    assert "examples/quickstart.yaml" in block
    for line in block.splitlines():
        if line.startswith("cp "):
            assert (ROOT / line.split()[1]).is_file(), line
        if line.startswith("rolescan "):
            command = line.split()[1]
            assert runner.invoke(app, [command, "--help"]).exit_code == 0, line
    assert "--dry" in block
    assert "--no-llm" in block
