"""The README's numbers: stated once, dated, and true.

Every one of these was wrong, repeated or unrecorded at some point: a test
count that was last true several releases ago, a ratio off by a factor of
three, a model run the evidence file contradicted. Prose cannot drift
unnoticed past a test that reads it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rolescan.cli import app

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_readme_hard_codes_no_test_count() -> None:
    assert not re.search(r"\b\d[\d,]*\s+tests\b", README), (
        "say 'over a thousand tests', not a number that goes stale"
    )


def test_the_prefilter_run_is_stated_once_with_the_right_ratio() -> None:
    assert README.count("526") == 1, "one dated paragraph, not three"
    assert README.count("93%") == 1
    assert round(526 / 36) == 15, "the ratio the README states"
    assert "fortieth" not in README
    assert "one in fifteen" in README


def test_the_ollama_figures_are_stated_once_and_scoped() -> None:
    assert README.count("72%") == 1
    assert README.count("50%") == 1
    paragraph = next(p for p in README.split("\n\n") if "72%" in p)
    assert "judge mode" in paragraph, "the figures belong to the older mode"
    assert "September 2026" in paragraph, "an undated measurement ages badly"


def test_the_readme_claims_no_unrecorded_model_run() -> None:
    lowered = README.lower()
    assert "have now run against a live server" not in lowered
    assert "anthropic backend scored" not in lowered
    assert "has since been revoked" not in lowered


@pytest.mark.parametrize(
    "heading",
    [
        "What gets hidden, and where to look",
        "Models, keys and cost",
        "Privacy and storage",
        "Public core and private plugins",
        "Measured quality",
    ],
)
def test_the_readme_has_the_section(heading: str) -> None:
    assert f"\n## {heading}\n" in README


def test_the_readme_points_at_the_generated_config_reference() -> None:
    assert "docs/config.md" in README
    assert (ROOT / "docs" / "config.md").is_file()


def test_every_command_the_readme_shows_exists() -> None:
    """`rolescan cvs` stayed in the command list for four releases after the
    command was removed."""
    blocks = "\n".join(re.findall(r"```[^\n]*\n(.*?)```", README, re.DOTALL))
    shown = set(re.findall(r"^rolescan (\w+)", blocks, re.MULTILINE))
    assert {"scan", "discover", "mark"} <= shown, "the pattern found too little"
    for command in sorted(shown):
        result = CliRunner().invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"README shows `rolescan {command}`"


def test_judge_mode_is_documented_as_supported_everywhere() -> None:
    """It was announced as "kept for one release, then removed" and then kept:
    an evaluation depends on it. No file may still promise its removal."""
    from rolescan.config import LLMConfig

    sources = {
        "README.md": README,
        "config.example.yaml": (ROOT / "config.example.yaml").read_text(),
        "docs/config.md": (ROOT / "docs" / "config.md").read_text(),
        "LLMConfig.mode": LLMConfig.model_fields["mode"].description or "",
    }
    for name, text in sources.items():
        flat = " ".join(text.split()).lower()
        assert "one release" not in flat, name
        assert "then removed" not in flat, name
    assert "\n**Judge mode.**" in README
    assert "llm.mode: judge" in README


def test_the_hidden_table_names_the_stats_clause_for_the_final_gate() -> None:
    """The row for `min_report_score` once said "Not listed" and nothing about
    the number the stats line gives."""
    row = next(
        line
        for line in README.splitlines()
        if line.startswith("| Its score was under `min_report_score`")
    )
    assert "N below min_report_score" in row


def test_no_row_advises_a_bare_dry_rerun_as_the_way_to_see_more() -> None:
    """A `--dry` re-run still drops every posting an earlier real scan
    recorded as seen, so "re-run with `--dry` and a lower number" finds
    nothing new after a real scan. The README says how to re-assess."""
    assert "Re-run with `--dry` and a lower number" not in README
    assert "still drops postings" in README
    section = README[README.index("## What gets hidden, and where to look") :]
    section = section[: section.index("\n## What you must supply")]
    assert "rolescan unsee URL" in section
    assert "output.db_path" in section


def test_the_readme_and_the_example_config_document_the_hidden_cap() -> None:
    assert "output.hidden_max" in README
    assert "-hidden.md" in README
    example = (ROOT / "config.example.yaml").read_text(encoding="utf-8")
    assert "hidden_max" in example


def test_the_readme_says_the_mail_servers_certificate_must_verify() -> None:
    section = README[README.index("- **Email.**") :]
    assert "certificate must verify" in section.split("- **What rolescan")[0]


def _cost_section() -> str:
    start = README.index("\n## Models, keys and cost\n")
    return README[start : README.index("\n## Privacy and storage\n")]


def test_the_cost_section_gives_the_one_measured_figure_and_no_other() -> None:
    """One recorded measurement: claude-haiku-4-5 in facts mode, two runs of 97
    postings on 8 October 2026 (USD 0.29 and USD 0.30), about USD 0.003 a
    posting. Any other price in this section would be an unrecorded claim."""
    section = _cost_section()
    assert "claude-haiku-4-5" in section and "facts mode" in section
    amounts = set(re.findall(r"USD\s?(\d+(?:\.\d+)?)", section))
    assert amounts == {"0.003", "0.29", "0.30"}
    assert "$" not in section
    assert "97 postings" in section and "8 October 2026" in section
    assert round(0.29 / 97, 3) == round(0.30 / 97, 3) == 0.003
    assert "none has been measured" not in section
