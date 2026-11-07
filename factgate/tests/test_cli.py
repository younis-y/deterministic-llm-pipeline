from pathlib import Path

from typer.testing import CliRunner

from factgate.cli import app

runner = CliRunner()


def test_init_then_replay_run(tmp_path: Path) -> None:
    folder = tmp_path / "starter"
    assert runner.invoke(app, ["init", str(folder)]).exit_code == 0
    config = folder / "config.yaml"
    result = runner.invoke(
        app,
        ["run", str(config), "--replay", str(folder / "answers.jsonl"), "--as-of", "2026-11-07"],
    )
    assert result.exit_code == 0, result.output
    assert "decided" in result.output
    assert "3 answers replayed from a file, no model called" in result.output
    lines = (folder / "out" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 3


def test_init_refuses_a_non_empty_folder(tmp_path: Path) -> None:
    (tmp_path / "x").write_text("keep me")
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code != 0 and "not empty" in result.output


def test_dry_makes_no_call(tmp_path: Path) -> None:
    folder = tmp_path / "s"
    runner.invoke(app, ["init", str(folder)])
    result = runner.invoke(app, ["run", str(folder / "config.yaml"), "--dry"])
    assert result.exit_code == 0, result.output
    assert "would send 3" in result.output
    assert not (folder / "out").exists()


def test_check_prints_prompt_and_schema(tmp_path: Path) -> None:
    folder = tmp_path / "s"
    runner.invoke(app, ["init", str(folder)])
    result = runner.invoke(app, ["check", str(folder / "config.yaml")])
    assert result.exit_code == 0
    assert "verbatim" in result.output and '"properties"' in result.output


def test_a_bad_config_fails_with_its_message(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: x\n")
    result = runner.invoke(app, ["check", str(path)])
    assert result.exit_code == 2 and "items" in result.output
